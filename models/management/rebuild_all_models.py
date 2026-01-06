import itertools
import subprocess
import sys
import os
import time

# Add models directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.utils.get_mlflow_uri import get_mlflow_uri

def main():
    # 1. Get MLflow URI
    mlflow_uri = get_mlflow_uri()
    print(f"Targeting MLflow at: {mlflow_uri}")

    # 2. Define Business Scenarios (Subset for validation, or full set)
    # We will use the full set but with strict 1 trial
    periods = [
        '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
    ]
    intervals = [
        '5m', '15m', '30m', '60m',  '1d', '1wk', '1mo'
    ]
    symbols = [
        'VALE3.SA', 'AAPL', 'NVDA', 'ITSA4.SA', 'WEGE3.SA', '^GSPC'
    ]
    
    # 3. Fixed "Solid" Hyperparameters for Rebuild
    # We don't want random search, we want one good model per scenario.
    default_params = {
        'framework': 'pytorch',
        'model_type': 'lstm',
        'sequence_length': '24',
        'batch_size': '32',
        'epochs': '30',        # Enough to learn something
        'learning_rate': '0.001',
        'hidden_units_1': '128',
        'hidden_units_2': '64',
        'dropout': '0.2',
        'num_layers': '2'
    }

    # Generate Business Scenarios
    business_scenarios = list(itertools.product(symbols, periods, intervals))
    total_scenarios = len(business_scenarios)
    
    print(f"Total Business Scenarios to Rebuild: {total_scenarios}")
    print("Starting Rebuild Sequence...")

    # Calculate absolute path to training script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, "../training/train_model_grid.py")

    start_time = time.time()
    
    def is_valid_combo(period, interval):
        # Approximation of YFinance Limits
        # 1m: max 7 days
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
            continue
            
        print(f"\n--- Rebuilding [{i+1}/{total_scenarios}]: {symbol} | {period} | {interval} ---")
        
        cmd = [
            "python3", training_script,
            "--mlflow_uri", mlflow_uri,
            "--symbol", symbol,
            "--period", period,
            "--interval", interval,
            "--framework", default_params['framework'],
            "--model_type", default_params['model_type'],
            "--sequence_length", default_params['sequence_length'],
            "--batch_size", default_params['batch_size'],
            "--learning_rate", default_params['learning_rate'],
            "--epochs", default_params['epochs'],
            "--hidden_units_1", default_params['hidden_units_1'],
            "--hidden_units_2", default_params['hidden_units_2'],
            "--dropout", default_params['dropout'],
            "--num_layers", default_params['num_layers'],
            "--feature_set", "full"
        ]
        
        try:
            # Run synchronously
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"  [Error] Rebuild failed for {symbol} {period} {interval}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\nRebuild Completed in {elapsed/60:.2f} minutes!")

if __name__ == "__main__":
    main()
