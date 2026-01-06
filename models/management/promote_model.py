import mlflow
import os
import sys
import logging
from mlflow.tracking import MlflowClient
from collections import defaultdict
import datetime

# Add models directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.utils.get_mlflow_uri import get_mlflow_uri

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPromoter:
    def __init__(self, metric="rmse", improvement_threshold=0.01):
        """
        Initialize the ModelPromoter.
        
        Args:
            metric (str): The metric to optimize (lower is better for rmse/mae).
            improvement_threshold (float): Minimum % improvement to overthrow a champion (0.01 = 1%).
        """
        self.mlflow_uri = get_mlflow_uri()
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient()
        self.metric = metric
        self.threshold = improvement_threshold
        logger.info(f"Connected to MLflow at {self.mlflow_uri}")

    def get_candidate_runs(self, experiment_id):
        """Fetch runs from an experiment, sorted by metric."""
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                order_by=[f"metrics.{self.metric} ASC"], # Assuming lower is better
                filter_string="status = 'FINISHED'"
            )
            return runs
        except Exception as e:
            logger.error(f"Error fetching runs for exp {experiment_id}: {e}")
            return []

    def get_registered_model(self, model_name):
        """Get registered model details or None."""
        try:
            return self.client.get_registered_model(model_name)
        except Exception:
            return None

    def get_latest_version_by_stage(self, model_name, stage):
        """Get the latest version in a specific stage."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            return versions[0] if versions else None
        except Exception:
            return None
            
    def get_version_by_alias(self, model_name, alias):
        """Get model version by alias."""
        try:
            return self.client.get_model_version_by_alias(model_name, alias)
        except Exception:
            return None

    def transition_and_tag(self, model_name, version, stage, alias, tags=None):
        """Helper to transition stage, set alias, and log tags."""
        logger.info(f"Promoting {model_name} v{version} to {stage} (Alias: @{alias})")
        
        # 1. Transition Stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production") # Only archive others if Production
        )
        
        # 2. Set Alias
        self.client.set_registered_model_alias(model_name, alias, version)
        
        # 3. Set Tags
        if tags:
            for k, v in tags.items():
                self.client.set_model_version_tag(model_name, version, k, str(v))

    def evaluate_and_promote(self, exp):
        """
        Core logic:
        1. Find best Run (Candidate).
        2. Find Current Production (Champion).
        3. Compare and Promote.
        """
        if not exp.name.startswith("Experiment_"):
            return

        logger.info(f"Processing Experiment: {exp.name}")
        
        # 1. Identify Model Name
        reg_model_name = exp.name.replace("Experiment_", "model_")
        
        # 2. Get Best Candidate Run
        runs = self.get_candidate_runs(exp.experiment_id)
        if not runs:
            logger.warning(f"No finished runs found for {exp.name}")
            return
            
        best_run = runs[0]
        candidate_metric = best_run.data.metrics.get(self.metric)
        candidate_run_id = best_run.info.run_id
        
        if candidate_metric is None:
            logger.warning(f"Best run {candidate_run_id} has no {self.metric} metric.")
            return

        logger.info(f"Best Candidate: Run {candidate_run_id} | {self.metric}: {candidate_metric:.4f}")

        # 3. Find Registered Version for this Candidate Run
        # We need to find the specific version created from this run
        all_versions = self.client.search_model_versions(f"name='{reg_model_name}'")
        candidate_version = next((v for v in all_versions if v.run_id == candidate_run_id), None)
        
        if not candidate_version:
            logger.warning(f"Run {candidate_run_id} is not registered as a version of {reg_model_name}. Skipping.")
            return

        # 4. Get Current Champion (Production) and Challenger (Staging)
        champion_version = self.get_latest_version_by_stage(reg_model_name, "Production")
        challenger_version = self.get_latest_version_by_stage(reg_model_name, "Staging")

        # SCENARIO A: No Champion Exists -> Candidate becomes Champion
        if not champion_version:
            logger.info(f"No Champion found for {reg_model_name}. Promoting Candidate to Production.")
            self.transition_and_tag(
                reg_model_name, candidate_version.version, "Production", "champion",
                tags={"promotion_reason": "init_champion", "promotion_date": datetime.datetime.now().isoformat()}
            )
            return

        # Fetch Champion Metric
        try:
            champion_run = self.client.get_run(champion_version.run_id)
            champion_metric = champion_run.data.metrics.get(self.metric)
        except Exception:
            logger.warning("Could not fetch Champion metrics. Assuming Candidate is better.")
            champion_metric = float('inf')

        logger.info(f"Current Champion: v{champion_version.version} | {self.metric}: {champion_metric:.4f}")

        # SCENARIO B: Compare Candidate vs Champion
        # Calculate Improvement: (Champion - Candidate) / Champion (assuming lower is better)
        # If unknown champion metric, improvement is infinite
        if champion_metric == 0:
             improvement_pct = 0 # Avoid div by zero
        else:
             improvement_pct = (champion_metric - candidate_metric) / champion_metric

        logger.info(f"Improvement: {improvement_pct:.2%} (Threshold: {self.threshold:.2%})")

        if improvement_pct > self.threshold:
            # Candidate Wins!
            logger.info(">>> Candidate DEFEATED Champion!")
            
            # 1. Old Champion -> Staging (Honorary Challenger)
            logger.info(f"Demoting Old Champion v{champion_version.version} to Staging (@challenger)")
            self.transition_and_tag(
                reg_model_name, champion_version.version, "Staging", "challenger",
                tags={"demotion_reason": "defeated_by_candidate", "demotion_date": datetime.datetime.now().isoformat()}
            )

            # 2. New Champion -> Production
            self.transition_and_tag(
                reg_model_name, candidate_version.version, "Production", "champion",
                tags={
                    "promotion_reason": "defeated_champion",
                    "improvement_pct": f"{improvement_pct:.2%}",
                    "beaten_version": champion_version.version,
                    "promotion_date": datetime.datetime.now().isoformat()
                }
            )
        
        else:
            # Candidate Lost against Champion
            logger.info("Candidate failed to beat Champion.")
            
            # Check if it should replace the current Challenger (Staging)
            # Evaluate against current Challenger if one exists
            promote_to_challenger = False
            
            if not challenger_version:
                promote_to_challenger = True
            else:
                # Compare against Challenger
                try:
                    challenger_run = self.client.get_run(challenger_version.run_id)
                    challenger_metric = challenger_run.data.metrics.get(self.metric)
                    if challenger_metric and candidate_metric < challenger_metric:
                         promote_to_challenger = True
                         logger.info(f"Candidate ({candidate_metric:.4f}) is better than current Challenger ({challenger_metric:.4f})")
                except:
                    promote_to_challenger = True # Default to replace if error
            
            if promote_to_challenger:
                if candidate_version.version == champion_version.version:
                    logger.info("Candidate is already the Champion. No action.")
                else:
                    logger.info("Promoting Candidate to Staging (@challenger)")
                    self.transition_and_tag(
                        reg_model_name, candidate_version.version, "Staging", "challenger",
                        tags={"promotion_reason": "best_alternative", "promotion_date": datetime.datetime.now().isoformat()}
                    )
            else:
                logger.info("Candidate is not better than current Challenger. No action.")

    def run_pipeline(self):
        """Run the full promotion pipeline."""
        logger.info("Starting Model Promotion Pipeline...")
        experiments = self.client.search_experiments()
        for exp in experiments:
            try:
                self.evaluate_and_promote(exp)
            except Exception as e:
                logger.error(f"Failed to process experiment {exp.name}: {e}")
        logger.info("Pipeline Finished.")

if __name__ == "__main__":
    promoter = ModelPromoter()
    promoter.run_pipeline()
