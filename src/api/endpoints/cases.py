"""API endpoints for case management."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from ...core.models import Case, AlertStatus
from ...core.auth import get_current_user

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("/", response_model=List[Dict[str, Any]])
async def list_cases(
    status: Optional[AlertStatus] = None,
    current_user: dict = Depends(get_current_user)
):
    """List cases with optional filtering."""
    # Mock implementation
    cases = [
        {
            "id": "case_1",
            "title": "High Value Transaction Alert",
            "status": "open",
            "priority": 3,
            "assignee": "analyst_1",
            "created_at": "2024-01-01T00:00:00Z"
        }
    ]
    
    if status:
        cases = [c for c in cases if c["status"] == status.value]
    
    return cases


@router.post("/", response_model=Dict[str, Any])
async def create_case(
    case: Case,
    current_user: dict = Depends(get_current_user)
):
    """Create a new case."""
    # Mock implementation
    return {
        "id": case.id,
        "message": "Case created successfully",
        "created_at": case.created_at.isoformat()
    }
