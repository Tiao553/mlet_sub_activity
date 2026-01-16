from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime

from app.core.logger import setup_logger
from app.services.fetcher import fetch_and_save_s3
from app.services.prediction import pipe_to_predict

router = APIRouter()
logger = setup_logger("prediction_router")

from app.schemas.prediction import StockInterval, StockPeriod, StockSymbol

@router.get("/stock-data-prediction")
def stock_data_endpoint(
    symbol: StockSymbol = Query(..., description="Stock symbol"),
    interval: StockInterval = Query(StockInterval.ONE_DAY, description="Data interval"),
    period: StockPeriod = Query(..., description="Data period (e.g. 1d, 5y)"),
):
    try:
        # Hardcoded defaults/logic since inputs were removed
        # Period is now the primary driver for data fetching
        start_date = None
        end_date = None
        auto_adjust = True
        
        # Convert Enums to strings for internal services
        symbol_str = symbol.value
        interval_str = interval.value
        period_str = period.value
        
        logger.info(f"Received prediction request for {symbol_str} (period={period_str}, interval={interval_str})")
        
        # Step 1: Fetch and Save Data to S3
        # Passing None for dates forces fetcher/yfinance to use 'period'
        msg, status_code = fetch_and_save_s3(symbol_str, start_date, end_date, interval_str, period_str, auto_adjust)
        
        if status_code != 200:
            logger.error(f"Failed to fetch data: {msg}")
            raise HTTPException(status_code=status_code, detail=msg)
            
        # Step 2: Run Prediction Pipeline
        prediction = pipe_to_predict(symbol_str, start_date, end_date, period_str, interval_str)
        
        if prediction is None:
            logger.warning("Prediction returned None")
            return {
                "symbol": symbol,
                "prediction": None,
                "message": "Could not generate prediction. Check logs/data availability."
            }
            
        logger.info(f"Prediction success for {symbol}: {prediction}")
        return {
            "symbol": symbol,
            "prediction": prediction,
            "message": "Success"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in stock_data_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
