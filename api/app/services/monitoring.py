from datetime import datetime
from app.core.logger import setup_logger
from app.core.config import settings
from app.services.s3 import write_json_to_s3

logger = setup_logger("monitoring_service")

def save_prediction_log(symbol: str, prediction_data: dict) -> None:
    """
    Saves prediction details to S3 for monitoring.
    Path: s3://<bucket>/predictions/<symbol>/<timestamp>.json
    """
    try:
        timestamp = datetime.now().isoformat()
        # Use a safe filename from timestamp
        filename = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        key = f"predictions/{symbol}/{filename}.json"
        
        # Enforce metadata
        payload = {
            "symbol": symbol,
            "timestamp": timestamp,
            "data": prediction_data,
            "model_version": "champion" # Placeholder if we don't have exact version yet
        }
        
        write_json_to_s3(settings.S3_BUCKET_NAME, key, payload)
        logger.info(f"Prediction log saved to {key}")
    except Exception as e:
        logger.error(f"Failed to save prediction log: {e}", exc_info=True)
        # We generally don't want monitoring failure to break the API response, so we catch it.
