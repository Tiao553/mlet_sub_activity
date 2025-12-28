import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- Configurações Pre-Import TensorFlow ---
# Desativa a compilação XLA e define flags antes de importar o TensorFlow
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Força CPU para evitar erros de JIT na GPU

import tensorflow as tf
tf.config.optimizer.set_jit(False)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import StochasticOscillator, AwesomeOscillatorIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.trend import CCIIndicator, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange

# --- Parâmetros fixos ---

symbol = 'VALE3.SA'
period = '7d'
interval = '1m'
SEQ_LENGTH_DEFAULT = 24
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

# --- Função para adicionar indicadores técnicos ---
def add_technical_indicators(df):
    if 'Close' not in df.columns:
        raise ValueError("DataFrame não contém coluna 'Close'")

    try:
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
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume'],
                window=14
            ).volume_weighted_average_price()
        else:
            df['VWAP'] = np.nan

        df['Candle_Body'] = df['Close'] - df['Open']
        df['Candle_Range'] = df['High'] - df['Low']
        df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']

    except Exception as e:
        print(f"Erro ao calcular indicadores técnicos: {str(e)}")
        raise

    return df

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc

# --- Função para criar sequências ---

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Close price na posição 0

    return np.array(X), np.array(y)

def train_and_evaluate(X_train, y_train, X_test, y_test, params, n_features):
    # 1. Limpar a sessão anterior para evitar acúmulo de modelos na memória da GPU
    tf.keras.backend.clear_session()
    
    # 2. Definição do modelo usando a nova API (Input Layer separada)
    model = Sequential([
        Input(shape=(params['seq_length'], n_features)),
        Bidirectional(LSTM(params['lstm_units_1'], return_sequences=True)),
        Dropout(params['dropout']),
        Bidirectional(LSTM(params['lstm_units_2'])),
        Dropout(params['dropout']),
        Dense(1)
    ])
    
    # 3. Compilação
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse')
    
    # 4. Early Stopping para evitar overfitting e economizar tempo
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=0
    )
    
    # 5. Treinamento
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.15, # Ajuste conforme sua variável VALIDATION_SPLIT
        verbose=0,
        callbacks=[early_stop]
    )
    
    # 6. Predições
    test_pred = model.predict(X_test, verbose=0)
    
    # 7. Função interna para inverter escala (mantida como a sua original)
    def inverse_transform(scaler, data):
        dummy = np.zeros((len(data), n_features))
        dummy[:, 0] = data.flatten()
        return scaler.inverse_transform(dummy)[:, 0]
    
    y_test_inv = inverse_transform(params['scaler'], y_test)
    test_pred_inv = inverse_transform(params['scaler'], test_pred)
    
    # 8. Métricas
    rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
    mae = mean_absolute_error(y_test_inv, test_pred_inv)
    
    # Limpeza forçada de lixo da memória RAM
    gc.collect()
    
    return rmse, mae, history, model

# --- Divisão treino/teste ---
def prepare_data(df, seq_length=SEQ_LENGTH_DEFAULT):
    num_samples = len(df) - seq_length - 1
    train_size = int(num_samples * (1 - TEST_SIZE))
    df_train = df.iloc[:train_size + seq_length + 1].copy()
    df_test = df.iloc[train_size:].copy()
    
    scaler = MinMaxScaler()
    scaler.fit(df_train)
    train_scaled = scaler.transform(df_train)
    test_scaled = scaler.transform(df_test)
    
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    return X_train, y_train, X_test, y_test, scaler

def main():
    df = yf.download(symbol, period=period, interval=interval, progress=True)
    df.columns = df.columns.droplevel(1)
    
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    
    available_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',             # Preços e volume básicos
        'RSI', 'Stoch_K', 'Stoch_D', 'Awesome_Oscillator',    # Indicadores de momentum
        'MACD', 'MACD_signal', 'MACD_diff',                   # MACD
        'CCI', 'ADX', 'ADX_pos', 'ADX_neg',                   # Indicadores de tendência
        'BB_upper', 'BB_middle', 'BB_lower',                  # Bandas de Bollinger
        'EMA_20',                                             # Média móvel exponencial
        'ATR',                                                # Volatilidade
        'VWAP', 'OBV', 'AccDistIndex',                        # Indicadores de volume
        'Candle_Body', 'Candle_Range', 'Upper_Shadow', 'Lower_Shadow'  # Candlestick features
    ]
    
    features = [f for f in available_features if f in df.columns]
    print("Features selecionadas:", features)
    
    df = df[features]

    # --- Espaço de hiperparâmetros para busca ---
    search_space = {
        'seq_length': [12, 24, 36, 48, 60],
        'batch_size': [16, 32, 64, 128],
        'epochs': [50, 80, 100, 120, 150],
        'lstm_units_1': [64, 128, 256, 384, 512],
        'lstm_units_2': [32, 64, 128, 192, 256],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]  
    }

    # --- Busca randomizada ---
    n_trials = 10  # quantas combinações testar
    best_rmse = float('inf')
    best_params = None
    best_model = None
    best_history = None
    results = []

    for trial in range(n_trials):
        params = {
            'seq_length': random.choice(search_space['seq_length']),
            'batch_size': random.choice(search_space['batch_size']),
            'epochs': random.choice(search_space['epochs']),
            'lstm_units_1': random.choice(search_space['lstm_units_1']),
            'lstm_units_2': random.choice(search_space['lstm_units_2']),
            'dropout': random.choice(search_space['dropout']),
            'learning_rate': random.choice(search_space['learning_rate']),
        }

        print(f"\nTrial {trial+1}/{n_trials} com params: {params}")

        # Preparar dados para o seq_length atual
        X_train, y_train, X_test, y_test, scaler = prepare_data(df, seq_length=params['seq_length'])
        params['scaler'] = scaler

        # Treinar e avaliar
        try:
            rmse, mae, history, model = train_and_evaluate(X_train, y_train, X_test, y_test, params, n_features=len(features))
            print(f"RMSE: {rmse:.4f} - MAE: {mae:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model
                best_history = history
            
            results.append({
                'trial': trial + 1,
                'rmse': rmse,
                'mae': mae,
                **params 
            })
            
        except Exception as e:
            print(f"Erro no treino/avaliação: {e}")
            # Não adicionamos aos resultados se falhou, ou adicionamos como erro
            results.append({
                'trial': trial + 1,
                'error': str(e),
                **params
            })

    print("\nMelhor RMSE:", best_rmse)
    print("Melhores hiperparâmetros:", best_params)

    # --- Plotar resultado com melhor modelo ---
    if best_model:
        X_train, y_train, X_test, y_test, scaler = prepare_data(df, seq_length=best_params['seq_length'])
        y_test_inv = scaler.inverse_transform(
            np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), len(features)-1))], axis=1)
        )[:, 0]

        test_pred = best_model.predict(X_test)
        test_pred_inv = scaler.inverse_transform(
            np.concatenate([test_pred, np.zeros((len(test_pred), len(features)-1))], axis=1)
        )[:, 0]

        plt.figure(figsize=(16, 8))
        plt.plot(y_test_inv, label='Valor Real')
        plt.plot(test_pred_inv, label='Previsão')
        plt.title(f'Previsão do Preço de Fechamento - {symbol} (Melhor Modelo)')
        plt.xlabel('Horas')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.savefig('prediction_plot.png')
        plt.close()
        print("Gráfico salvo como 'prediction_plot.png'")

    # Salvando o melhor modelo em formato HDF5 (.h5)
    if best_model:
        best_model.save('../model/modelo_v1.h5')
        print("Modelo salvo com sucesso no arquivo 'best_lstm_model.h5'")

if __name__ == "__main__":
    main()
