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

    # 2. Define Parameter Grid
    # This grid covers the "Switch Case" requirements (Periods/Intervals)
    # and optimizes model architecture.
    
    # Business Parameters (The "Switch Case" Keys)
    periods = ['7d', '1mo']
    intervals = ['1m', '15m']
    symbols = ['VALE3.SA']
    
    # Model/Training Parameters (The Search Space)
    frameworks = ['tensorflow', 'pytorch']
    model_types = ['lstm', 'bi_lstm']
    sequence_lengths = [24, 60]
    batch_sizes = [32]
    learning_rates = [0.001]
    epochs_list = [20] # Keep low for demo/speed, increase for real
    
    # Cartesian Product
    combinations = list(itertools.product(
        periods, intervals, symbols, frameworks, model_types, sequence_lengths, batch_sizes, learning_rates, epochs_list
    ))
    
    print(f"Total combinations to run: {len(combinations)}")
    
    for i, combo in enumerate(combinations):
        period, interval, symbol, framework, model_type, seq_len, batch, lr, epochs = combo
        
        print(f"\n--- Running [{i+1}/{len(combinations)}] ---")
        print(f"Combo: {combo}")
        
        # Calculate absolute path to training script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        training_script = os.path.join(script_dir, "../training/train_model_grid.py")
        
        cmd = [
            "python3", training_script,
            "--mlflow_uri", mlflow_uri,
            "--symbol", symbol,
            "--period", period,
            "--interval", interval,
            "--framework", framework,
            "--model_type", model_type,
            "--sequence_length", str(seq_len),
            "--batch_size", str(batch),
            "--learning_rate", str(lr),
            "--epochs", str(epochs),
            "--feature_set", "full"
        ]
        
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Run failed: {e}")
            # Continue to next run even if one fails
            continue

    print("\nGrid Search Completed!")

if __name__ == "__main__":
    main()
