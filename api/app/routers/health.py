from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "app_name": settings.PROJECT_NAME,
        "message": "Service is healthy"
    }
