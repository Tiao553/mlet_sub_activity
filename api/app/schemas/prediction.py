from pydantic import BaseModel
from typing import Optional
from enum import Enum

class StockInterval(str, Enum):
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    SIXTY_MINUTES = "60m"
    ONE_DAY = "1d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"

class StockPeriod(str, Enum):
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    MAX = "max"

class StockSymbol(str, Enum):
    VALE3 = "VALE3.SA"
    AAPL = "AAPL"
    NVDA = "NVDA"
    ITSA4 = "ITSA4.SA"
    WEGE3 = "WEGE3.SA"
    GSPC = "^GSPC"

class PredictionRequest(BaseModel):
    symbol: StockSymbol
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: StockInterval = StockInterval.ONE_DAY
    period: Optional[StockPeriod] = None
    auto_adjust: bool = True

class PredictionResponse(BaseModel):
    symbol: str
    prediction: Optional[float]
    last_processed_date: Optional[str] = None
    message: Optional[str] = None
