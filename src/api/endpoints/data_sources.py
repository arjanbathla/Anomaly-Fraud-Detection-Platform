"""API endpoints for data source management."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from ...core.models import DataSource, DataSourceType
from ...core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-sources", tags=["data-sources"])


@router.get("/", response_model=List[Dict[str, Any]])
async def list_data_sources(current_user: dict = Depends(get_current_user)):
    """List all data sources."""
    # Mock implementation - in production, this would query the database
    return [
        {
            "id": "ds_1",
            "name": "Credit Card Transactions",
            "source_type": "kafka",
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        },
        {
            "id": "ds_2", 
            "name": "Network Logs",
            "source_type": "csv",
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        }
    ]


@router.post("/", response_model=Dict[str, Any])
async def create_data_source(
    data_source: DataSource,
    current_user: dict = Depends(get_current_user)
):
    """Create a new data source."""
    # Check permissions
    if current_user.get("role") not in ["data_engineer", "admin"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Mock implementation
    return {
        "id": data_source.id,
        "message": "Data source created successfully",
        "created_at": data_source.created_at.isoformat()
    }


@router.get("/{source_id}", response_model=Dict[str, Any])
async def get_data_source(
    source_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific data source."""
    # Mock implementation
    return {
        "id": source_id,
        "name": "Sample Data Source",
        "source_type": "kafka",
        "config": {"topic": "transactions"},
        "is_active": True
    }
