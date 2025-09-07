"""Authentication manager for the platform."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from ..config.settings import settings
from ..exceptions import AuthenticationError, AuthorizationError
from ..models import User, UserRole

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()


class AuthManager:
    """Manages user authentication and JWT tokens."""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise AuthenticationError("Could not validate credentials")
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        # In a real implementation, this would query a database
        # For now, we'll use a simple in-memory user store
        users = {
            "admin": User(
                username="admin",
                email="admin@example.com",
                role=UserRole.ADMIN,
                password_hash=self.get_password_hash("admin123")
            ),
            "analyst": User(
                username="analyst",
                email="analyst@example.com", 
                role=UserRole.ANALYST,
                password_hash=self.get_password_hash("analyst123")
            ),
            "data_engineer": User(
                username="data_engineer",
                email="data_engineer@example.com",
                role=UserRole.DATA_ENGINEER,
                password_hash=self.get_password_hash("data123")
            ),
            "model_owner": User(
                username="model_owner",
                email="model_owner@example.com",
                role=UserRole.MODEL_OWNER,
                password_hash=self.get_password_hash("model123")
            )
        }
        
        user = users.get(username)
        if not user:
            return None
        
        if not self.verify_password(password, user.password_hash):
            return None
        
        return user


# Global auth manager instance
auth_manager = AuthManager()


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token."""
    return auth_manager.create_access_token(data, expires_delta)


def verify_token(token: str) -> Dict[str, Any]:
    """Verify a token."""
    return auth_manager.verify_token(token)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get the current authenticated user."""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Could not validate credentials")
        
        # In a real implementation, this would query the database
        # For now, return the user info from the token
        return {
            "username": username,
            "role": payload.get("role"),
            "email": payload.get("email")
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
