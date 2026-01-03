import itertools
import subprocess
import sys
import os
import time
import random

# Add models directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.utils.get_mlflow_uri import get_mlflow_uri

def main():
    # 1. Get MLflow URI
    mlflow_uri = get_mlflow_uri()
    print(f"Targeting MLflow at: {mlflow_uri}")

    # 2. Define Business Scenarios (Switch Case Keys)
    # The combination of these defines the "Business Requirements"
    periods = [
        '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
    ]
    intervals = [
        '5m', '15m', '30m', '60m',  '1d', '1wk', '1mo'
    ]
    symbols = [
        'VALE3.SA', 'AAPL', 'NVDA', 'ITSA4.SA', 'WEGE3.SA', '^GSPC'
    ]

    # 3. Define Hyperparameter Search Space (from reference yluiiyyb)
    search_space = {
        'framework': ['tensorflow', 'pytorch'],
        'model_type': ['lstm', 'bi_lstm', 'gru'],
        'sequence_length': [12, 24, 36, 48, 60],
        'batch_size': [16, 32, 64, 128],
        'epochs': [20, 50, 80],
        'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
        'hidden_units_1': [64, 128, 256, 384, 512],
        'hidden_units_2': [32, 64, 128, 192, 256],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'num_layers': [1, 2]
    }

    # Configuration
    N_TRIALS_PER_SCENARIO = 5 # Number of random trials per Symbol/Period/Interval combo
    
    # Generate Business Scenarios
    business_scenarios = list(itertools.product(symbols, periods, intervals))
    total_scenarios = len(business_scenarios)
    
    print(f"Total Business Scenarios: {total_scenarios}")
    print(f"Total Runs (Approx): {total_scenarios * N_TRIALS_PER_SCENARIO}")
    print("Starting Random Search...")

    # Calculate absolute path to training script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, "../training/train_model_grid.py")

    start_time = time.time()
    
    import random # Ensure random is available

    def is_valid_combo(period, interval):
        # Approximation of YFinance Limits
        # 1m: max 7 days (technically 30d possible but often restricted per request)
        if interval == '1m':
            if period not in ['1d', '5d', '7d']:
                return False
        
        # 2m - 30m: max 60 days
        if interval in ['2m', '5m', '15m', '30m', '90m']:
            if period in ['3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
                 return False
                 
        # 1h: max 730 days (2y)
        if interval == '1h':
            if period in ['5y', '10y', 'max']:
                return False
                
        return True

    for i, (symbol, period, interval) in enumerate(business_scenarios):
        if not is_valid_combo(period, interval):
            print(f"--- Skipping Scenario [{i+1}/{total_scenarios}]: {symbol} | {period} | {interval} (Invalid Combo) ---")
            continue
            
        print(f"\n--- Scenario [{i+1}/{total_scenarios}]: {symbol} | {period} | {interval} ---")
        
        for trial in range(N_TRIALS_PER_SCENARIO):
            # Sample Random Hyperparameters
            params = {k: random.choice(v) for k, v in search_space.items()}
            
            print(f"  > Trial {trial+1}/{N_TRIALS_PER_SCENARIO}: {params}")
            
            cmd = [
                "python3", training_script,
                "--mlflow_uri", mlflow_uri,
                "--symbol", symbol,
                "--period", period,
                "--interval", interval,
                "--framework", params['framework'],
                "--model_type", params['model_type'],
                "--sequence_length", str(params['sequence_length']),
                "--batch_size", str(params['batch_size']),
                "--learning_rate", str(params['learning_rate']),
                "--epochs", str(params['epochs']),
                "--hidden_units_1", str(params['hidden_units_1']),
                "--hidden_units_2", str(params['hidden_units_2']),
                "--dropout", str(params['dropout']),
                "--num_layers", str(params['num_layers']),
                "--feature_set", "full"
            ]
            
            try:
                # Run synchronously to avoid overloading machine
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                # Train script handles data errors (1m/10y) by printing warning and exiting.
                # If it exits with non-zero, it means a crash.
                # We log and continue.
                print(f"  [Error] Run failed: {e}")
                continue

    elapsed = time.time() - start_time
    print(f"\nRandom Search Completed in {elapsed/60:.2f} minutes!")

if __name__ == "__main__":
    main()
