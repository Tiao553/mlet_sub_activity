import os
import sys

import pandas as pd

import mlflow

# Add models directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models.utils.get_mlflow_uri import get_mlflow_uri


def generate_report(
    experiment_name="Grid_Search_Experiment",
    output_file="docs/model_selection_report.md",
):
    mlflow_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    print(f"Connecting to MLflow at {mlflow_uri}...")

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment {experiment_name} not found.")
            return

        # Fetch all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            print("No runs found.")
            return

        # Prepare Markdown
        md_content = f"# Model Selection Report\n\n"
        md_content += f"**Generated Date:** {pd.Timestamp.now()}\n"
        md_content += f"**MLflow Server:** [{mlflow_uri}]({mlflow_uri})\n\n"

        md_content += "## Best Models by Scenario (Switch Case Keys)\n"
        md_content += "This section identifies the best performing model for each unique combination of **Symbol**, **Period**, and **Interval**.\n\n"

        # Group by Business Keys
        # Ensure columns exist (tags are prefixed with 'tags.')
        group_cols = ["tags.symbol", "tags.period", "tags.interval"]
        available_cols = [c for c in group_cols if c in runs.columns]

        if not available_cols:
            print("Missing business tags in runs.")
            return

        grouped = runs.groupby(available_cols)

        for name, group in grouped:
            # Name is a tuple of values corresponding to group_cols
            symbol, period, interval = name

            # Find best run (lowest RMSE)
            best_run = group.sort_values("metrics.rmse").iloc[0]

            run_id = best_run.run_id
            rmse = best_run["metrics.rmse"]
            mae = best_run["metrics.mae"]
            framework = best_run["tags.framework"]
            artifact_uri = best_run.artifact_uri

            # Extract relevant params (cols start with 'params.')
            param_cols = [c for c in runs.columns if c.startswith("params.")]
            params = {c.replace("params.", ""): best_run[c] for c in param_cols}

            md_content += f"### Scenario: {symbol} | {period} | {interval}\n"
            md_content += f"- **Selected Model ID:** `{run_id}`\n"
            md_content += f"- **Framework:** {framework}\n"
            md_content += f"- **RMSE:** {rmse:.4f} (Best in Group)\n"
            md_content += f"- **MAE:** {mae:.4f}\n"
            md_content += f"- **Key Parameters:**\n"
            md_content += f"  - Model Type: {params.get('model_type')}\n"
            md_content += f"  - Layers: {params.get('num_layers')}\n"
            md_content += f"  - Units: {params.get('hidden_units_1')}\n"
            md_content += f"- **Justification:** Achieved lowest error metric among {len(group)} trials for this configuration.\n"
            md_content += f"- **Artifact Access:** [Link]({artifact_uri.replace('s3://', 'http://s3-browser-link/')}) (Requires S3 access or MLflow UI)\n"
            md_content += f"- **MLflow Run Link:** [{run_id}]({mlflow_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id})\n\n"

        # Write to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(md_content)

        print(f"Report generated at {output_file}")

    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    generate_report()
