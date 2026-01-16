import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Any, Dict
# from tensorflow.keras.models import load_model # Removed to avoid heavy dependency in Lambda
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
from ta.trend import MACD, CCIIndicator, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.volume import VolumeWeightedAveragePrice
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from app.services.s3 import read_csv_from_s3
from app.services.monitoring import save_prediction_log
from app.core.logger import setup_logger
from app.core.config import settings

logger = setup_logger("prediction_service")

# Cache structure: {model_uri: {"model": model_obj, "params": dict}}
MODEL_CACHE: Dict[str, Dict] = {}

def get_model_and_params(symbol: str, period: str, interval: str):
    """
    Loads model and its training parameters from MLflow.
    Returns dict: {"model": model, "params": {seq_length, feature_set, ...}}
    """
    p = period if period else "1y"
    i = interval if interval else "1d"
    
    # Using 'champion' alias
    model_name = f"model_{symbol}_{p}_{i}"
    alias = "champion"
    model_uri = f"models:/{model_name}@{alias}"
    
    if model_uri in MODEL_CACHE:
        return MODEL_CACHE[model_uri]
        
    try:
        logger.info(f"Loading model details for {model_uri}")
        
        # 1. Get Run ID to fetch params
        client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        mv = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(mv.run_id)
        params = run.data.params
        
        # Parse critical params
        seq_length = int(params.get("sequence_length", 24))
        feature_set = params.get("feature_set", "full")
        
        logger.info(f"Model params resolved: seq_length={seq_length}, feature_set={feature_set}")
        
        # 2. Load Model
        logger.info(f"Loading MLflow model: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        cache_entry = {
            "model": model,
            "params": {
                "sequence_length": seq_length,
                "feature_set": feature_set
            }
        }
        MODEL_CACHE[model_uri] = cache_entry
        return cache_entry
        
    except Exception as e:
        logger.error(f"Failed to load model {model_uri}: {e}")
        return None

def create_sequences(data, seq_length):
    X = []
    # We only need the LAST sequence for prediction
    # If data has N rows, we need rows [N-seq_length : N]
    if len(data) < seq_length:
        return np.array([])
        
    # Take the last chunk
    last_seq = data[-seq_length:]
    X.append(last_seq)
    return np.array(X)

def add_technical_indicators(df: pd.DataFrame, feature_set: str = "full") -> pd.DataFrame:
    try:
        # Standardize columns
        # df.columns are already likely standard from fetcher, but ensure Title Case
        # Note: yfinance often returns 'Close', 'High', etc.
        
        if 'Close' not in df.columns:
            # Fallback if columns are lowercase
            df.columns = df.columns.str.title()
            
        if 'Close' not in df.columns:
             raise ValueError("DataFrame missing 'Close' column")

        # Basic Indicators (Always present according to training script)
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        
        if feature_set == "full":
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['Stoch_K'] = stoch.stoch()
            # Note: training script MIGHT treat Stoch_D differently or not check it? 
            # Looking at reproduction script I wrote: training features included 'Stoch_K'.
            # Let's add them all to be safe, selection happens later.
            
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            
            bb = BollingerBands(close=df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            # df['BB_lower'] = bb.bollinger_lband() # Training script 'full' sets features: BB_upper. Wait. 
            # Re-verify training script features list:
            # features = ['Close', 'High', 'Low', 'Volume', 'RSI', 'EMA_20']
            # if full: extend(['Stoch_K', 'MACD', 'BB_upper', 'ATR', 'OBV'])
            # So purely these 11.
            
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
            df['ATR'] = atr.average_true_range()
            
            obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
            df['OBV'] = obv.on_balance_volume()

        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.exception(f"Erro ao adicionar indicadores técnicos: {e}")
        raise

def pipe_to_predict(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> Optional[float]:
    try:
        # 1. Resolve Model First (to know requirements)
        model_entry = get_model_and_params(symbol, period, interval)
        if not model_entry:
            logger.error(f"Não foi possível carregar modelo e parâmetros para {symbol}")
            return None
            
        model = model_entry["model"]
        seq_length = model_entry["params"]["sequence_length"]
        feature_set_name = model_entry["params"]["feature_set"]
        
        logger.info(f"Model resolved. Seq Len: {seq_length}, Features: {feature_set_name}")

        # 2. Read Data
        logger.info(f"Lendo dados do S3 para o símbolo: {symbol}")
        data = read_csv_from_s3(settings.S3_BUCKET_NAME, f"fetch/{symbol}_evolution.csv")
        if data is None or data.empty:
            logger.error("Nenhum dado encontrado no S3.")
            return None
        
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data.sort_values("datetime", inplace=True)
        
        df = data.copy() # Use full data for scaling context

        # 3. Preprocessing
        df = add_technical_indicators(df, feature_set_name)
        
        # 4. Feature Selection
        # Must match training script EXACTLY
        features_list = ['Close', 'High', 'Low', 'Volume', 'RSI', 'EMA_20']
        if feature_set_name == "full":
            features_list.extend(['Stoch_K', 'MACD', 'BB_upper', 'ATR', 'OBV'])
            
        # Verify columns exist
        missing_cols = [c for c in features_list if c not in df.columns]
        if missing_cols:
            logger.error(f"Missing features in data: {missing_cols}")
            return None
            
        df_features = df[features_list].copy()
        
        # 5. Scaling
        # We must fit scaler on current data. 
        # Ideally, we would load the scaler artifact from training. 
        # As fallback, fitting on latest 500+ points is a reasonable approximation for trend-following models.
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_features.values)
        
        # 6. Sequence Creation
        if len(scaled_data) < seq_length:
            logger.error(f"Insufficient data ({len(scaled_data)}) for sequence length {seq_length}")
            return None
            
        last_sequence = scaled_data[-seq_length:]
        last_sequence = last_sequence.reshape(1, seq_length, len(features_list))
        
        # Cast to float32 (Torch requirement)
        last_sequence = last_sequence.astype(np.float32)

        # 7. Predict
        logger.info(f"Predicting w/ shape {last_sequence.shape}...")
        prediction_scaled = model.predict(last_sequence)
        
        # Extract value
        if isinstance(prediction_scaled, pd.DataFrame):
             pred_val = prediction_scaled.iloc[0,0]
        elif isinstance(prediction_scaled, np.ndarray):
             pred_val = prediction_scaled.flat[0]
        elif isinstance(prediction_scaled, list):
             pred_val = prediction_scaled[0]
        else:
             pred_val = float(prediction_scaled)
             
        # 8. Inverse Transform
        # Reconstruct a dummy row to use scaler.inverse_transform
        # We assume 'Close' is at index 0 (as per features_list)
        dummy = np.zeros((1, len(features_list)))
        dummy[0, 0] = pred_val 
        
        # Note: Inverse transform applies scale factors. 
        # Since we only populated the target column, others are 0 (which is fake but doesn't affect target if scaler is feature-wise independent).
        # MinMaxScaler IS feature-wise independent.
        pred_inverse = scaler.inverse_transform(dummy)[0, 0]
        
        logger.info(f"Predicted Scaled: {pred_val:.4f} -> Inverse: {pred_inverse:.4f}")

        # Save monitoring
        save_prediction_log(symbol, {
            "period": period,
            "interval": interval,
            "input_features_shape": last_sequence.shape,
            "predicted_value": float(pred_inverse),
            "last_close": float(df['Close'].iloc[-1])
        })
            
        return float(pred_inverse)
    
    except Exception as e:
        logger.exception(f"Erro na execução do pipeline de predição: {e}")
        return None
