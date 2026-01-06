import argparse
import os
import random
import numpy as np

# --- Configurações Pre-Import TensorFlow (Replicated from working script) ---
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Força CPU para evitar erros de JIT na GPU

import mlflow
import mlflow.tensorflow
import mlflow.pytorch
import torch
import tensorflow as tf
tf.config.optimizer.set_jit(False)

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
from ta.trend import MACD, CCIIndicator, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Grid Search Training Script")
    
    # Data Parameters
    parser.add_argument("--symbol", type=str, default="VALE3.SA", help="Stock symbol")
    parser.add_argument("--period", type=str, default="7d", help="Data period")
    parser.add_argument("--interval", type=str, default="1m", help="Data interval")
    parser.add_argument("--sequence_length", type=int, default=24, help="Input sequence length")
    parser.add_argument("--feature_set", type=str, default="full", choices=["basic", "full"], help="Feature set to use")

    # Model Parameters
    parser.add_argument("--framework", type=str, default="pytorch", choices=["tensorflow", "pytorch"], help="DL Framework")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "gru", "bi_lstm"], help="Model architecture")
    parser.add_argument("--hidden_units_1", type=int, default=128, help="Units in first layer")
    parser.add_argument("--hidden_units_2", type=int, default=64, help="Units in second layer")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked layers (if processed)")

    # Training Parameters
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--loss_function", type=str, default="mse", choices=["mse", "mae"], help="Loss function")
    
    # MLflow
    parser.add_argument("--mlflow_uri", type=str, required=True, help="MLflow Tracking URI")
    parser.add_argument("--experiment_name", type=str, default="Grid_Search_Experiment", help="MLflow Experiment Name")

    return parser.parse_args()

# --- Data Preparation ---
def add_technical_indicators(df, feature_set="full"):
    if 'Close' not in df.columns:
        raise ValueError("DataFrame missing 'Close' column")
    
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()

    if feature_set == "full":
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        
        bb = BollingerBands(close=df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
        df['ATR'] = atr.average_true_range()
        
        obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
        df['OBV'] = obv.on_balance_volume()

    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0]) # Assuming Close is at index 0
    return np.array(X), np.array(y)

# --- TensorFlow Model ---
def build_tf_model(input_shape, args):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    if args.model_type == "bi_lstm":
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.hidden_units_1, return_sequences=True)))
        model.add(tf.keras.layers.Dropout(args.dropout))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.hidden_units_2)))
    elif args.model_type == "gru":
        model.add(tf.keras.layers.GRU(args.hidden_units_1, return_sequences=True))
        model.add(tf.keras.layers.Dropout(args.dropout))
        model.add(tf.keras.layers.GRU(args.hidden_units_2))
    else: # LSTM
        model.add(tf.keras.layers.LSTM(args.hidden_units_1, return_sequences=True))
        model.add(tf.keras.layers.Dropout(args.dropout))
        model.add(tf.keras.layers.LSTM(args.hidden_units_2))
        
    model.add(tf.keras.layers.Dropout(args.dropout))
    model.add(tf.keras.layers.Dense(1))
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate) if args.optimizer == "adam" else tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
    model.compile(optimizer=opt, loss=args.loss_function)
    return model

# --- PyTorch Model ---
class PyTorchModel(torch.nn.Module):
    def __init__(self, input_size, args):
        super(PyTorchModel, self).__init__()
        self.hidden_size = args.hidden_units_1
        self.num_layers = args.num_layers
        
        if args.model_type == "gru":
             self.rnn = torch.nn.GRU(input_size, args.hidden_units_1, num_layers=args.num_layers, batch_first=True, dropout=args.dropout if args.num_layers > 1 else 0)
        elif args.model_type == "bi_lstm":
             self.rnn = torch.nn.LSTM(input_size, args.hidden_units_1, num_layers=args.num_layers, batch_first=True, bidirectional=True, dropout=args.dropout if args.num_layers > 1 else 0)
        else:
             self.rnn = torch.nn.LSTM(input_size, args.hidden_units_1, num_layers=args.num_layers, batch_first=True, dropout=args.dropout if args.num_layers > 1 else 0)
             
        fc_input_dim = args.hidden_units_1 * 2 if args.model_type == "bi_lstm" else args.hidden_units_1
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :] # Last time step
        out = self.fc(out)
        return out

# --- Main ---
def main():
    args = parse_args()
    
    # 1. Setup MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    
    # Dynamic Experiment Name based on Business Keys
    experiment_name = f"Experiment_{args.symbol}_{args.period}_{args.interval}"
    mlflow.set_experiment(experiment_name)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.symbol}_{args.period}_{args.interval}_{args.framework}_{timestamp}"

    # START RUN EARLY to capture everything
    with mlflow.start_run(run_name=run_name) as run:
        # Log all args immediately
        for arg, value in vars(args).items():
            mlflow.log_param(arg, value)
            
        # Log Business Tags
        mlflow.set_tag("symbol", args.symbol)
        mlflow.set_tag("period", args.period)
        mlflow.set_tag("interval", args.interval)
        mlflow.set_tag("framework", args.framework)
        mlflow.set_tag("environment", "dev")

        print(f"Downloading {args.symbol}...")
        try:
            df = yf.download(args.symbol, period=args.period, interval=args.interval, progress=False)
        except Exception as e:
            print(f"Error downloading data: {e}")
            mlflow.set_tag("param_trace", "download_error")
            return

        # Check if data was returned
        if df.empty:
             msg = f"No data returned for {args.symbol} ({args.period}/{args.interval})"
             print(f"WARNING: {msg}. Skipping.")
             mlflow.set_tag("skip_reason", "empty_data_from_api")
             return

        # Fix for multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        # Check if data is sufficient for indicators (Window=26 is max for MACD, plus some buffer)
        if len(df) < 35:
            msg = f"Data length ({len(df)}) too short for technical indicators (min 35)"
            print(f"WARNING: {msg}. Skipping {args.symbol} {args.period}/{args.interval}.")
            mlflow.set_tag("skip_reason", "insufficient_data_length")
            mlflow.log_param("data_length", len(df))
            return

        try:
            df = add_technical_indicators(df, args.feature_set)
        except Exception as e:
            print(f"Indicator Error: {e}")
            mlflow.set_tag("skip_reason", "indicator_calculation_error")
            return
        
        # Check if data survived indicator generation (dropna)
        min_required_len = args.sequence_length + 10 # Need at least seq_len + some data for split
        if len(df) < min_required_len:
            msg = f"Insufficient data after indicators ({len(df)} < {min_required_len})"
            print(f"WARNING: {msg} for {args.symbol}. Skipping.")
            mlflow.set_tag("skip_reason", "insufficient_data_after_preprocessing")
            mlflow.log_param("processed_len", len(df))
            return

        # Select Features
        features = ['Close', 'High', 'Low', 'Volume', 'RSI', 'EMA_20']
        if args.feature_set == "full":
            features.extend(['Stoch_K', 'MACD', 'BB_upper', 'ATR', 'OBV'])
        
        # Ensure features exist
        features = [f for f in features if f in df.columns]
        print(f"Features: {features}")
        
        dataset = df[features].values
        
        # Check for NaNs/Infs
        if np.isnan(dataset).any() or np.isinf(dataset).any():
             print("WARNING: Dataset contains NaNs or Infs after pre-processing. filling with 0")
             dataset = np.nan_to_num(dataset)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataset)
        
        X, y = create_sequences(scaled_data, args.sequence_length)
        
        if len(X) == 0:
            print(f"WARNING: No sequences created for {args.symbol} (len={len(df)}, seq_len={args.sequence_length}). Skipping.")
            mlflow.set_tag("skip_reason", "no_sequences_created")
            return

        # Train/Test Split
        split = int(len(X) * 0.8)
        
        # Ensure there is at least 1 train and 1 test sample
        if split == 0 or split == len(X):
            print(f"WARNING: Insufficient samples for split ({len(X)} samples). Skipping.")
            mlflow.set_tag("skip_reason", "insufficient_samples_for_split")
            return

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Registered Model Name uses the same business keys
        reg_model_name = f"model_{args.symbol}_{args.period}_{args.interval}"

        # Train & Evaluate
        rmse, mae = 0, 0
        
        if args.framework == "tensorflow":
            try:
                # Pre-config for TF (Already done globally)
                
                model = build_tf_model((X_train.shape[1], X_train.shape[2]), args)
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, 
                                    validation_split=0.2, callbacks=[early_stop], verbose=1)
                
                preds = model.predict(X_test)
                
                # Inverse Transform
                dummy = np.zeros((len(preds), len(features)))
                dummy[:, 0] = preds.flatten()
                preds_inv = scaler.inverse_transform(dummy)[:, 0]
                
                dummy_y = np.zeros((len(y_test), len(features)))
                dummy_y[:, 0] = y_test.flatten()
                y_test_inv = scaler.inverse_transform(dummy_y)[:, 0]
                
                rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
                mae = mean_absolute_error(y_test_inv, preds_inv)
                
                try:
                    # 1. Log Model (Artifacts to S3)
                    model_info = mlflow.tensorflow.log_model(
                        model, 
                        "model"
                    )
                    
                    # 2. Register Model (Metadata to DB)
                    try:
                        mlflow.register_model(
                            model_uri=model_info.model_uri,
                            name=reg_model_name
                        )
                    except Exception as reg_error:
                         print(f"Warning: Failed to Register TF model: {reg_error}")

                except Exception as e:
                    print(f"Warning: Failed to log TF model artifact: {e}")
                
            except Exception as e:
                print(f"TF Error: {e}")
                mlflow.log_param("error", str(e))
                # Don't raise, just log error so next trial continues
                pass

        elif args.framework == "pytorch":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = PyTorchModel(X_train.shape[2], args).to(device)
            criterion = torch.nn.MSELoss() if args.loss_function == "mse" else torch.nn.L1Loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) if args.optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=args.learning_rate)
            
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            
            # Simple Training Loop
            model.train()
            for epoch in range(args.epochs):
                optimizer.zero_grad()
                out = model(X_train_t)
                loss = criterion(out, y_train_t)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).cpu().numpy()
                
            # Inverse Transform
            dummy = np.zeros((len(preds), len(features)))
            dummy[:, 0] = preds.flatten()
            preds_inv = scaler.inverse_transform(dummy)[:, 0]
            
            dummy_y = np.zeros((len(y_test), len(features)))
            dummy_y[:, 0] = y_test.flatten()
            y_test_inv = scaler.inverse_transform(dummy_y)[:, 0]
            
            rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
            mae = mean_absolute_error(y_test_inv, preds_inv)
            
            try:
                # 1. Log Model (Artifacts to S3)
                model_info = mlflow.pytorch.log_model(
                    model, 
                    "model"
                )
                
                # 2. Register Model (Metadata to DB)
                try:
                    mlflow.register_model(
                        model_uri=model_info.model_uri,
                        name=reg_model_name
                    )
                except Exception as reg_error:
                        print(f"Warning: Failed to Register PyTorch model: {reg_error}")
                        
            except Exception as e:
                print(f"Warning: Failed to log PyTorch model artifact: {e}")

        print(f"Finished Run. RMSE: {rmse}, MAE: {mae}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # Save Plot
        plt.figure(figsize=(10,6))
        plt.plot(y_test_inv, label="True")
        plt.plot(preds_inv, label="Pred")
        plt.title(f"{args.symbol} Prediction ({args.framework})")
        plt.legend()
        plot_path = "prediction.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

if __name__ == "__main__":
    main()
