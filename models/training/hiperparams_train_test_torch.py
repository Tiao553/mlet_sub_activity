import gc
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import (AwesomeOscillatorIndicator, RSIIndicator,
                         StochasticOscillator)
from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import (AccDistIndexIndicator, OnBalanceVolumeIndicator,
                       VolumeWeightedAveragePrice)
from torch.utils.data import DataLoader, TensorDataset

# --- Configurações de Hardware ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- Parâmetros fixos ---
symbol = "VALE3.SA"
period = "7d"
interval = "1m"
SEQ_LENGTH_DEFAULT = 24
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2


# --- Função para adicionar indicadores técnicos (Mantida idêntica) ---
def add_technical_indicators(df):
    if "Close" not in df.columns:
        raise ValueError("DataFrame não contém coluna 'Close'")

    try:
        df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

        stoch = StochasticOscillator(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            window=14,
            smooth_window=3,
        )
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()

        ao = AwesomeOscillatorIndicator(
            high=df["High"], low=df["Low"], window1=5, window2=34
        )
        df["Awesome_Oscillator"] = ao.awesome_oscillator()

        macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        df["CCI"] = CCIIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], window=20
        ).cci()

        adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
        df["ADX"] = adx.adx()
        df["ADX_pos"] = adx.adx_pos()
        df["ADX_neg"] = adx.adx_neg()

        df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()

        bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_middle"] = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()

        df["ATR"] = AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        ).average_true_range()

        df["OBV"] = OnBalanceVolumeIndicator(
            close=df["Close"], volume=df["Volume"]
        ).on_balance_volume()
        df["AccDistIndex"] = AccDistIndexIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
        ).acc_dist_index()

        if all(col in df.columns for col in ["High", "Low", "Close", "Volume"]):
            df["VWAP"] = VolumeWeightedAveragePrice(
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                volume=df["Volume"],
                window=14,
            ).volume_weighted_average_price()
        else:
            df["VWAP"] = np.nan

        df["Candle_Body"] = df["Close"] - df["Open"]
        df["Candle_Range"] = df["High"] - df["Low"]
        df["Upper_Shadow"] = df["High"] - df[["Close", "Open"]].max(axis=1)
        df["Lower_Shadow"] = df[["Close", "Open"]].min(axis=1) - df["Low"]

    except Exception as e:
        print(f"Erro ao calcular indicadores técnicos: {str(e)}")
        raise

    return df


# --- Modelo LSTM em PyTorch ---
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_rate):
        super(TimeSeriesLSTM, self).__init__()
        # LSTM 1: Bidirectional
        self.lstm1 = nn.LSTM(
            input_size, hidden_size_1, batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        # LSTM 2: Bidirectional (input size doubles because of bidirectional previous layer)
        self.lstm2 = nn.LSTM(
            hidden_size_1 * 2, hidden_size_2, batch_first=True, bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output Layer (taking the last hidden state of bidirectional lstm2)
        self.fc = nn.Linear(hidden_size_2 * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        # out shape: (batch_size, seq_len, hidden_size_1 * 2)
        # We only care about the last output for the sequence-to-one prediction,
        # BUT the original Keras code had 'return_sequences=True' for the first layer
        # and default (False) for the second. So we pass the full sequence to the second LSTM.

        out, (h_n, c_n) = self.lstm2(out)
        out = self.dropout2(out)

        # out shape: (batch_size, seq_len, hidden_size_2 * 2)
        # We take the output of the last time step
        out = out[:, -1, :]

        out = self.fc(out)
        return out


# --- Função para criar sequências ---
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i : (i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Close price na posição 0
    return np.array(X), np.array(y)


# --- Função de Treino e Avaliação ---
def train_and_evaluate(X_train, y_train, X_test, y_test, params, n_features):
    # Converter para Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Criar DataLoaders para batch training
    # Dividir treino em treino/validação para Early Stopping
    val_size = int(len(X_train_tensor) * 0.15)
    train_size = len(X_train_tensor) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        TensorDataset(X_train_tensor, y_train_tensor), [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    # Instanciar Modelo
    model = TimeSeriesLSTM(
        input_size=n_features,
        hidden_size_1=params["lstm_units_1"],
        hidden_size_2=params["lstm_units_2"],
        dropout_rate=params["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Early Stopping Variables
    dignity_patience = 10
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    history = {"loss": [], "val_loss": []}

    # Training Loop
    for epoch in range(params["epochs"]):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_dataset)
        history["loss"].append(epoch_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                running_val_loss += loss.item() * X_val.size(0)

        val_loss = running_val_loss / len(val_dataset)
        history["val_loss"].append(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= dignity_patience:
                # print(f"Early stop at epoch {epoch}")
                break

    # Carregar melhor modelo
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Predições no Teste
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor).cpu().numpy()

    y_test_cpu = y_test  # já é numpy

    # Inverter escala
    def inverse_transform(scaler, data):
        dummy = np.zeros((len(data), n_features))
        dummy[:, 0] = data.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    y_test_inv = inverse_transform(params["scaler"], y_test_cpu)
    test_pred_inv = inverse_transform(params["scaler"], test_pred)

    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
    mae = mean_absolute_error(y_test_inv, test_pred_inv)

    return rmse, mae, history, model


# --- Prepare Data ---
def prepare_data(df, seq_length=SEQ_LENGTH_DEFAULT):
    num_samples = len(df) - seq_length - 1
    train_size = int(num_samples * (1 - TEST_SIZE))
    df_train = df.iloc[: train_size + seq_length + 1].copy()
    df_test = df.iloc[train_size:].copy()

    scaler = MinMaxScaler()
    scaler.fit(df_train)
    train_scaled = scaler.transform(df_train)
    test_scaled = scaler.transform(df_test)

    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    return X_train, y_train, X_test, y_test, scaler


def main():
    # Carregar dados
    df = yf.download(symbol, period=period, interval=interval, progress=True)
    df.columns = df.columns.droplevel(1)

    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    available_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "RSI",
        "Stoch_K",
        "Stoch_D",
        "Awesome_Oscillator",
        "MACD",
        "MACD_signal",
        "MACD_diff",
        "CCI",
        "ADX",
        "ADX_pos",
        "ADX_neg",
        "BB_upper",
        "BB_middle",
        "BB_lower",
        "EMA_20",
        "ATR",
        "VWAP",
        "OBV",
        "AccDistIndex",
        "Candle_Body",
        "Candle_Range",
        "Upper_Shadow",
        "Lower_Shadow",
    ]

    features = [f for f in available_features if f in df.columns]
    print("Features selecionadas:", features)
    df = df[features]

    # Espaço de busca
    search_space = {
        "seq_length": [12, 24, 36, 48, 60],
        "batch_size": [16, 32, 64, 128],
        "epochs": [50, 80, 100, 120, 150],
        "lstm_units_1": [64, 128, 256, 384, 512],
        "lstm_units_2": [32, 64, 128, 192, 256],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
    }

    n_trials = 10
    best_rmse = float("inf")
    best_params = None
    best_model = None
    results = []

    for trial in range(n_trials):
        params = {
            "seq_length": random.choice(search_space["seq_length"]),
            "batch_size": random.choice(search_space["batch_size"]),
            "epochs": random.choice(search_space["epochs"]),
            "lstm_units_1": random.choice(search_space["lstm_units_1"]),
            "lstm_units_2": random.choice(search_space["lstm_units_2"]),
            "dropout": random.choice(search_space["dropout"]),
            "learning_rate": random.choice(search_space["learning_rate"]),
        }

        print(f"\nTrial {trial+1}/{n_trials} com params: {params}")

        X_train, y_train, X_test, y_test, scaler = prepare_data(
            df, seq_length=params["seq_length"]
        )
        params["scaler"] = scaler

        try:
            rmse, mae, _, model = train_and_evaluate(
                X_train, y_train, X_test, y_test, params, n_features=len(features)
            )
            print(f"RMSE: {rmse:.4f} - MAE: {mae:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model

            results.append({"trial": trial + 1, "rmse": rmse, "mae": mae, **params})

        except Exception as e:
            print(f"Erro no treino/avaliação: {e}")
            import traceback

            traceback.print_exc()

        # Clean GPU memory
        torch.cuda.empty_cache()
        gc.collect()

    print("\nMelhor RMSE:", best_rmse)
    print("Melhores hiperparâmetros:", best_params)

    # Plotar resultado
    if best_model:
        X_train, y_train, X_test, y_test, scaler = prepare_data(
            df, seq_length=best_params["seq_length"]
        )

        best_model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            test_pred = best_model(X_test_tensor).cpu().numpy()

        y_test_inv = scaler.inverse_transform(
            np.concatenate(
                [y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))],
                axis=1,
            )
        )[:, 0]

        test_pred_inv = scaler.inverse_transform(
            np.concatenate(
                [test_pred, np.zeros((len(test_pred), len(features) - 1))], axis=1
            )
        )[:, 0]

        plt.figure(figsize=(16, 8))
        plt.plot(y_test_inv, label="Valor Real")
        plt.plot(test_pred_inv, label="Previsão")
        plt.title(f"Previsão do Preço de Fechamento - {symbol} (Melhor Modelo PyTorch)")
        plt.xlabel("Horas")
        plt.ylabel("Preço (R$)")
        plt.legend()
        plt.savefig("prediction_plot_torch.png")
        plt.close()
        print("Gráfico salvo como 'prediction_plot_torch.png'")

        # Salvar modelo
        torch.save(best_model.state_dict(), "../model/modelo_v1.pth")
        print("Modelo salvo em '../model/modelo_v1.pth'")


if __name__ == "__main__":
    main()
