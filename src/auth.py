import os
import jwt
import datetime
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
EXPIRE_MINUTES = 60 * 24  #the token expire after 24h

security = HTTPBearer()


def create_access_token(user_id: str, email: str, is_admin: bool) -> str:
    """
    Creates a JWT access token for the given user information.
    """
    payload = {
        "sub": user_id,
        "email": email,
        "is_admin": is_admin,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decodes a JWT access token and returns the payload.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expirée, reconnecte-toi")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Dépendance FastAPI : extract and check the token sent in the header
    'Authorization: Bearer <token>'. Used on each protected endpoint.
    """
    return decode_access_token(credentials.credentials)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """
    Raise an exception if the token does not have the is_admin=True claim.
    """
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Accès réservé aux administrateurs")
    return user