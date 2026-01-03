import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from tensorflow.keras.models import load_model
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
from ta.trend import MACD, CCIIndicator, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.volume import VolumeWeightedAveragePrice
import mlflow.pyfunc

from app.services.s3_utils import read_csv_from_s3
from app.config.logger import setup_logger

logger = setup_logger("predictor")

# Caminho do modelo
# Caminho do modelo
BUCKET_NAME = "tech-challanger-4-prd-raw-zone-593793061865"
SEQ_LENGTH = 24 

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Close price na posição 0
    return np.array(X), np.array(y)

MODEL_CACHE = {}

# --- Função para carregar e prever ---
def get_model_dynamic(symbol: str, period: str, interval: str):
    """
    Carrega o modelo do MLflow baseado nos parÃ¢metros.
    Nome do modelo: model_{symbol}_{period}_{interval}
    Alias: @champion
    """
    # Defaults caso venha None
    # (Ajuste conforme a lógica de negócio ou garanta que a API passe valores válidos)
    p = period if period else "1y"
    i = interval if interval else "1d"
    
    # Sanitização básica (ex: remover .SA se o nome do modelo não usar, mas assumindo que usa)
    model_name = f"model_{symbol}_{p}_{i}"
    model_uri = f"models:/{model_name}@champion"
    
    if model_uri in MODEL_CACHE:
        return MODEL_CACHE[model_uri]
        
    try:
        logger.info(f"Tentando carregar modelo do MLflow: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        MODEL_CACHE[model_uri] = model
        logger.info("Modelo carregado e cacheado com sucesso.")
        return model
    except Exception as e:
        logger.error(f"Falha ao carregar modelo {model_uri}: {e}")
        return None

def predict_next_price(model, data: np.ndarray) -> Optional[float]:
    try:
        # MLflow pyfunc predict espera array ou dataframe
        prediction = model.predict(data)
        
        # Tratamento de retorno (pode ser array, lista, ou dataframe)
        if isinstance(prediction, pd.DataFrame):
            val = prediction.iloc[0, 0]
        elif isinstance(prediction, np.ndarray):
            val = prediction.flat[0]
        elif isinstance(prediction, list):
            val = prediction[0]
        else:
            val = float(prediction)
            
        return float(val)

    except Exception as e:
        logger.exception(f"Erro durante a predição: {e}")
    return None

# --- Função para indicadores técnicos ---
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.columns = df.columns.str.title()
        if 'Close' not in df.columns:
            raise ValueError("DataFrame não contém a coluna 'Close'.")

        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        ao = AwesomeOscillatorIndicator(high=df['High'], low=df['Low'], window1=5, window2=34)
        df['Awesome_Oscillator'] = ao.awesome_oscillator()
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['ADX_pos'] = adx.adx_pos()
        df['ADX_neg'] = adx.adx_neg()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        df['AccDistIndex'] = AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).acc_dist_index()
        if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
            df['VWAP'] = VolumeWeightedAveragePrice(
                high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14
            ).volume_weighted_average_price()
        else:
            df['VWAP'] = np.nan
        df['Candle_Body'] = df['Close'] - df['Open']
        df['Candle_Range'] = df['High'] - df['Low']
        df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        return df
    except Exception as e:
        logger.exception(f"Erro ao adicionar indicadores técnicos: {e}")
        raise

# --- Pipeline principal ---
def pipe_to_predict(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> Optional[float]:
    try:
        logger.info(f"Lendo dados do S3 para o símbolo: {symbol}")
        data = read_csv_from_s3(BUCKET_NAME, f"fetch/{symbol}_evolution.csv")
        if data is None or data.empty:
            logger.error("Nenhum dado encontrado no S3.")
            return None
        
        # Converte datetime de forma robusta, sem forçar o formato
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data.sort_values("datetime", inplace=True)

        # Conversão das datas de entrada
        if start_date:
            start_date = pd.to_datetime(start_date)
        else:
            start_date = data['datetime'].min()

        if end_date:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = data['datetime'].max()

        df = data
        #df = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

        if df.empty:
            logger.warning("Nenhum dado disponível no intervalo de tempo fornecido.")
            return None

        df = add_technical_indicators(df)
        df.dropna(inplace=True)

        available_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'Stoch_K', 'Stoch_D', 'Awesome_Oscillator',
            'MACD', 'MACD_signal', 'MACD_diff',
            'CCI', 'ADX', 'ADX_pos', 'ADX_neg',
            'BB_upper', 'BB_middle', 'BB_lower',
            'EMA_20', 'ATR', 'VWAP', 'OBV', 'AccDistIndex',
            'Candle_Body', 'Candle_Range', 'Upper_Shadow', 'Lower_Shadow'
        ]

        features = [f for f in available_features if f in df.columns]
        if not features:
            logger.error("Nenhuma feature válida disponível para predição.")
            return None

        if len(df) < SEQ_LENGTH + 1:
            logger.error(f"Dados insuficientes para criar uma sequência de tamanho {SEQ_LENGTH}.")
            return None

        # Extraindo as features e convertendo para numpy
        data_for_model = df[features].to_numpy()

        # Criar sequências
        X, y = create_sequences(data_for_model, SEQ_LENGTH)

        # Pega a última sequência para previsão
        last_sequence = X[-1].reshape(1, SEQ_LENGTH, len(features))

        # Carregar modelo dinamicamente
        model = get_model_dynamic(symbol, period, interval)
        if not model:
            # Fallback ou erro
            logger.error(f"Não foi possível carregar modelo para {symbol} period={period} interval={interval}")
            return None

        logger.info(f"Realizando predição com modelo MLflow. shape {last_sequence.shape}.")
        return predict_next_price(model, last_sequence)
    
    except Exception as e:
        logger.exception(f"Erro na execução do pipeline de predição: {e}")
        return None

# --- Exemplo de chamada ---
if __name__ == "__main__":
    symbol = "TSLA"
    valor_previsto = pipe_to_predict(symbol, start_date="2025-06-06", end_date="2025-06-06")
    print(f"Próximo valor previsto para {symbol}: {valor_previsto}")
