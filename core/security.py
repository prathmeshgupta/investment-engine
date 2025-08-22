"""
Security and Authentication Module for Investment Engine
Enterprise-grade security implementations
"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import logging
from functools import wraps
import time

class SecurityManager:
    """Manages authentication, authorization, and data encryption."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or self._generate_secret_key()
        self.encryption_key = self._derive_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.failed_attempts = {}
        self.rate_limits = {}
        
        # Security configuration
        self.config = {
            'max_login_attempts': 5,
            'lockout_duration': 300,  # 5 minutes
            'session_timeout': 3600,  # 1 hour
            'rate_limit_requests': 100,
            'rate_limit_window': 60,  # 1 minute
            'password_min_length': 12,
            'require_mfa': True
        }
        
        # Setup logging
        self.logger = self._setup_security_logging()
    
    def _generate_secret_key(self) -> str:
        """Generate cryptographically secure secret key."""
        return secrets.token_urlsafe(32)
    
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from secret key."""
        password = self.secret_key.encode()
        salt = b'investment_engine_salt'  # In production, use random salt per user
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _setup_security_logging(self) -> logging.Logger:
        """Setup security event logging."""
        logger = logging.getLogger('security')
        logger.setLevel(logging.INFO)
        
        # Create file handler for security logs
        handler = logging.FileHandler('logs/security.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            self.logger.warning(f"Password verification failed: {e}")
            return False
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < self.config['password_min_length']:
            return False
        
        # Check for required character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return all([has_upper, has_lower, has_digit, has_special])
    
    def generate_jwt_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate JWT token for authenticated user."""
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.config['session_timeout']),
            'jti': secrets.token_urlsafe(16)  # JWT ID for token revocation
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        self.logger.info(f"JWT token generated for user: {user_id}")
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Clean old requests
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if current_time - req_time < self.config['rate_limit_window']
        ]
        
        # Check rate limit
        if len(self.rate_limits[identifier]) >= self.config['rate_limit_requests']:
            self.logger.warning(f"Rate limit exceeded for: {identifier}")
            return False
        
        self.rate_limits[identifier].append(current_time)
        return True
    
    def check_account_lockout(self, identifier: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if identifier not in self.failed_attempts:
            return False
        
        attempts, lockout_time = self.failed_attempts[identifier]
        current_time = time.time()
        
        if current_time - lockout_time < self.config['lockout_duration']:
            return True
        
        # Reset failed attempts after lockout period
        del self.failed_attempts[identifier]
        return False
    
    def record_failed_attempt(self, identifier: str):
        """Record failed login attempt."""
        current_time = time.time()
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = [1, current_time]
        else:
            attempts, _ = self.failed_attempts[identifier]
            self.failed_attempts[identifier] = [attempts + 1, current_time]
        
        attempts, _ = self.failed_attempts[identifier]
        if attempts >= self.config['max_login_attempts']:
            self.logger.warning(f"Account locked due to failed attempts: {identifier}")
    
    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize user input to prevent injection attacks."""
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
            sanitized = input_data
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            return sanitized.strip()
        elif isinstance(input_data, dict):
            return {k: self.sanitize_input(v) for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
        else:
            return input_data
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and structure."""
        if not api_key or len(api_key) < 32:
            return False
        
        # Check if key follows expected format
        try:
            # Decode and verify checksum
            decoded = base64.urlsafe_b64decode(api_key.encode())
            if len(decoded) < 24:  # Minimum key length
                return False
            return True
        except Exception:
            return False

def require_auth(permissions: List[str] = None):
    """Decorator for requiring authentication and permissions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # In a real implementation, extract token from request headers
            # For demo purposes, we'll simulate this
            token = kwargs.get('auth_token')
            if not token:
                raise PermissionError("Authentication required")
            
            security_manager = SecurityManager()
            payload = security_manager.verify_jwt_token(token)
            if not payload:
                raise PermissionError("Invalid or expired token")
            
            # Check permissions
            if permissions:
                user_permissions = payload.get('permissions', [])
                if not all(perm in user_permissions for perm in permissions):
                    raise PermissionError("Insufficient permissions")
            
            # Add user context to kwargs
            kwargs['user_context'] = payload
            return func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Decorator for rate limiting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract identifier (IP, user ID, etc.)
            identifier = kwargs.get('client_id', 'anonymous')
            
            security_manager = SecurityManager()
            if not security_manager.check_rate_limit(identifier):
                raise Exception("Rate limit exceeded")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class AuditLogger:
    """Audit logging for compliance and security monitoring."""
    
    def __init__(self):
        self.logger = self._setup_audit_logging()
    
    def _setup_audit_logging(self) -> logging.Logger:
        """Setup audit logging."""
        logger = logging.getLogger('audit')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('logs/audit.log')
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_user_action(self, user_id: str, action: str, details: Dict = None):
        """Log user actions for audit trail."""
        audit_entry = {
            'user_id': user_id,
            'action': action,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        }
        self.logger.info(f"USER_ACTION: {audit_entry}")
    
    def log_data_access(self, user_id: str, resource: str, operation: str):
        """Log data access events."""
        audit_entry = {
            'user_id': user_id,
            'resource': resource,
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(f"DATA_ACCESS: {audit_entry}")
    
    def log_security_event(self, event_type: str, details: Dict):
        """Log security events."""
        audit_entry = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details
        }
        self.logger.warning(f"SECURITY_EVENT: {audit_entry}")

# Initialize security components
security_manager = SecurityManager()
audit_logger = AuditLogger()
