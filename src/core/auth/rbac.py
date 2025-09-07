"""Role-based access control (RBAC) for the platform."""

from typing import Dict, List, Set
from enum import Enum
import logging

from ..models import UserRole

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """System permissions."""
    # Data sources
    CREATE_DATA_SOURCE = "create_data_source"
    READ_DATA_SOURCE = "read_data_source"
    UPDATE_DATA_SOURCE = "update_data_source"
    DELETE_DATA_SOURCE = "delete_data_source"
    
    # Models
    CREATE_MODEL = "create_model"
    READ_MODEL = "read_model"
    UPDATE_MODEL = "update_model"
    DELETE_MODEL = "delete_model"
    PROMOTE_MODEL = "promote_model"
    
    # Scoring
    SCORE_RECORDS = "score_records"
    
    # Alerts
    READ_ALERTS = "read_alerts"
    UPDATE_ALERTS = "update_alerts"
    
    # Cases
    CREATE_CASE = "create_case"
    READ_CASE = "read_case"
    UPDATE_CASE = "update_case"
    
    # Analytics
    READ_ANALYTICS = "read_analytics"
    
    # Admin
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"


class RBACManager:
    """Manages role-based access control."""
    
    def __init__(self):
        self.role_permissions = self._initialize_role_permissions()
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize role-permission mappings."""
        return {
            UserRole.ADMIN: {
                Permission.CREATE_DATA_SOURCE,
                Permission.READ_DATA_SOURCE,
                Permission.UPDATE_DATA_SOURCE,
                Permission.DELETE_DATA_SOURCE,
                Permission.CREATE_MODEL,
                Permission.READ_MODEL,
                Permission.UPDATE_MODEL,
                Permission.DELETE_MODEL,
                Permission.PROMOTE_MODEL,
                Permission.SCORE_RECORDS,
                Permission.READ_ALERTS,
                Permission.UPDATE_ALERTS,
                Permission.CREATE_CASE,
                Permission.READ_CASE,
                Permission.UPDATE_CASE,
                Permission.READ_ANALYTICS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_SYSTEM,
            },
            UserRole.MODEL_OWNER: {
                Permission.CREATE_MODEL,
                Permission.READ_MODEL,
                Permission.UPDATE_MODEL,
                Permission.DELETE_MODEL,
                Permission.PROMOTE_MODEL,
                Permission.SCORE_RECORDS,
                Permission.READ_ALERTS,
                Permission.READ_ANALYTICS,
            },
            UserRole.DATA_ENGINEER: {
                Permission.CREATE_DATA_SOURCE,
                Permission.READ_DATA_SOURCE,
                Permission.UPDATE_DATA_SOURCE,
                Permission.DELETE_DATA_SOURCE,
                Permission.READ_MODEL,
                Permission.SCORE_RECORDS,
                Permission.READ_ALERTS,
                Permission.READ_ANALYTICS,
            },
            UserRole.ANALYST: {
                Permission.READ_DATA_SOURCE,
                Permission.READ_MODEL,
                Permission.SCORE_RECORDS,
                Permission.READ_ALERTS,
                Permission.UPDATE_ALERTS,
                Permission.CREATE_CASE,
                Permission.READ_CASE,
                Permission.UPDATE_CASE,
                Permission.READ_ANALYTICS,
            },
        }
    
    def has_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """Check if a role has a specific permission."""
        return permission in self.role_permissions.get(user_role, set())
    
    def get_user_permissions(self, user_role: UserRole) -> Set[Permission]:
        """Get all permissions for a user role."""
        return self.role_permissions.get(user_role, set())
    
    def check_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """Check permission and log the result."""
        has_perm = self.has_permission(user_role, permission)
        
        if not has_perm:
            logger.warning(f"Access denied: Role {user_role} does not have permission {permission}")
        
        return has_perm


# Global RBAC manager instance
rbac_manager = RBACManager()


def check_permission(user_role: UserRole, permission: Permission) -> bool:
    """Check if a user role has a specific permission."""
    return rbac_manager.check_permission(user_role, permission)
