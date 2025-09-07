"""API endpoints for alert management."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from ...core.models import Alert, AlertStatus
from ...core.auth import get_current_user

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/", response_model=List[Dict[str, Any]])
async def list_alerts(
    status: Optional[AlertStatus] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """List alerts with optional filtering."""
    # Mock implementation
    alerts = [
        {
            "id": "alert_1",
            "entity_id": "user_123",
            "score": 0.95,
            "status": "open",
            "created_at": "2024-01-01T00:00:00Z"
        }
    ]
    
    if status:
        alerts = [a for a in alerts if a["status"] == status.value]
    
    return alerts[:limit]


@router.put("/{alert_id}/status", response_model=Dict[str, Any])
async def update_alert_status(
    alert_id: str,
    status: AlertStatus,
    current_user: dict = Depends(get_current_user)
):
    """Update alert status."""
    # Mock implementation
    return {
        "id": alert_id,
        "status": status.value,
        "updated_by": current_user["username"],
        "message": "Alert status updated successfully"
    }
