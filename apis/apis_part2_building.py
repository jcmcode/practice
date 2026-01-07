"""
================================================================================
APIS IN PYTHON - COMPREHENSIVE CHEAT SHEET (PART 2: BUILDING APIS)
================================================================================
Building REST APIs with Flask and FastAPI, request/response handling,
authentication, error handling, database integration, and deployment.

CONTENTS:
1. Flask Basics: routing, request/response, blueprints
2. FastAPI Fundamentals: type hints, automatic docs, validation
3. Path & Query Parameters: URL patterns, type validation
4. Request Bodies: JSON parsing, form data, file uploads
5. Response Handling: custom status codes, headers, streaming
6. Error Handling: exceptions, error responses, logging
7. Middleware & Decorators: request/response hooks
8. Authentication: JWT, OAuth2, role-based access
9. Database Integration: SQLAlchemy, migrations, relationships
10. Testing APIs: pytest, fixtures, client testing
11. API Documentation: OpenAPI/Swagger, custom docs
12. Rate Limiting & Throttling: request limits, backoff
13. CORS & Security: cross-origin, headers, CSRF
14. WebSockets: real-time communication
15. Deployment: containerization, cloud platforms
16. Monitoring & Logging: structured logs, metrics
17. Best Practices: versioning, consistency, error patterns

================================================================================
"""

import json
import time
import logging
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import jwt
import hashlib


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. FLASK BASICS
# ============================================================================

def flask_hello_world():
    """Basic Flask app with simple routes"""
    try:
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        
        # Route with GET method (default)
        @app.route("/", methods=["GET"])
        def index():
            return jsonify({"message": "Hello, World!"})
        
        # Route with path parameter
        @app.route("/users/<int:user_id>", methods=["GET"])
        def get_user(user_id):
            return jsonify({"id": user_id, "name": "John Doe"})
        
        # Route with multiple methods
        @app.route("/items", methods=["GET", "POST"])
        def items():
            if request.method == "POST":
                data = request.get_json()
                return jsonify({"created": data}), 201
            else:
                return jsonify({"items": []})
        
        # Query parameters
        @app.route("/search", methods=["GET"])
        def search():
            query = request.args.get("q", "")
            limit = request.args.get("limit", 10, type=int)
            return jsonify({"query": query, "limit": limit})
        
        # Return custom status code
        @app.route("/created", methods=["POST"])
        def create():
            return jsonify({"id": 1}), 201  # 201 Created
        
        # Return with custom headers
        @app.route("/custom-headers", methods=["GET"])
        def custom_headers():
            return (
                jsonify({"data": "value"}),
                200,
                {"X-Custom-Header": "custom-value"}
            )
        
        # if __name__ == "__main__":
        #     app.run(debug=True, port=5000)
        
    except ImportError:
        print("flask not installed; pip install flask")


def flask_blueprints():
    """Organize routes with blueprints"""
    try:
        from flask import Flask, Blueprint, jsonify
        
        app = Flask(__name__)
        
        # Create blueprint for users
        users_bp = Blueprint("users", __name__, url_prefix="/api/users")
        
        @users_bp.route("", methods=["GET"])
        def list_users():
            return jsonify({"users": []})
        
        @users_bp.route("/<int:user_id>", methods=["GET"])
        def get_user(user_id):
            return jsonify({"id": user_id})
        
        @users_bp.route("", methods=["POST"])
        def create_user():
            return jsonify({"id": 1}), 201
        
        # Create blueprint for posts
        posts_bp = Blueprint("posts", __name__, url_prefix="/api/posts")
        
        @posts_bp.route("", methods=["GET"])
        def list_posts():
            return jsonify({"posts": []})
        
        # Register blueprints
        app.register_blueprint(users_bp)
        app.register_blueprint(posts_bp)
        
    except ImportError:
        print("flask not installed")


# ============================================================================
# 2. FASTAPI FUNDAMENTALS
# ============================================================================

def fastapi_hello_world():
    """Basic FastAPI app with automatic docs"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        
        app = FastAPI(
            title="My API",
            description="A great API",
            version="1.0.0"
        )
        
        # GET with path parameter
        @app.get("/")
        async def root():
            return {"message": "Hello, World!"}
        
        # Path parameter with type validation
        @app.get("/users/{user_id}")
        async def get_user(user_id: int):
            # Type hint ensures user_id is int, validated automatically
            return {"id": user_id, "name": "John"}
        
        # Query parameter
        @app.get("/search")
        async def search(q: str, limit: int = 10):
            # q is required, limit defaults to 10
            return {"query": q, "limit": limit}
        
        # Optional query parameter
        @app.get("/filter")
        async def filter_items(skip: int = 0, take: Optional[int] = None):
            return {"skip": skip, "take": take}
        
        # Request body with Pydantic model
        class Item(BaseModel):
            name: str
            price: float
            description: Optional[str] = None
        
        @app.post("/items", status_code=201)
        async def create_item(item: Item):
            return {"id": 1, **item.dict()}
        
        # Uvicorn: run with: uvicorn apis_part2:app --reload
        
    except ImportError:
        print("fastapi not installed; pip install fastapi uvicorn")


# ============================================================================
# 3. PYDANTIC MODELS & VALIDATION
# ============================================================================

def pydantic_models_and_validation():
    """Input validation with Pydantic"""
    try:
        from pydantic import BaseModel, Field, validator, root_validator
        from typing import Optional, List
        
        # Simple model
        class User(BaseModel):
            id: int
            username: str
            email: str
        
        # Model with defaults
        class Post(BaseModel):
            title: str
            content: str
            published: bool = False
            views: int = 0
        
        # Field with constraints
        class Product(BaseModel):
            name: str = Field(..., min_length=1, max_length=100)
            price: float = Field(..., gt=0)  # price > 0
            quantity: int = Field(default=1, ge=0)  # quantity >= 0
            description: Optional[str] = None
        
        # Custom validators
        class Account(BaseModel):
            username: str
            email: str
            
            @validator("username")
            def username_alphanumeric(cls, v):
                if not v.isalnum():
                    raise ValueError("must be alphanumeric")
                return v
            
            @validator("email")
            def email_valid(cls, v):
                if "@" not in v:
                    raise ValueError("invalid email")
                return v
        
        # Root validator (validate multiple fields together)
        class PasswordChange(BaseModel):
            old_password: str
            new_password: str
            new_password_confirm: str
            
            @root_validator
            def passwords_match(cls, values):
                if values.get("new_password") != values.get("new_password_confirm"):
                    raise ValueError("passwords don't match")
                return values
        
        # Nested models
        class Address(BaseModel):
            street: str
            city: str
            country: str
        
        class UserWithAddress(BaseModel):
            name: str
            address: Address  # Nested validation
        
        # List validation
        class TeamCreate(BaseModel):
            name: str
            members: List[str] = Field(..., min_items=1, max_items=50)
        
        # Usage
        product = Product(name="Widget", price=9.99)
        print(product.dict())
        
        # Config for model behavior
        class Config(BaseModel):
            name: str
            
            class Config:
                validate_assignment = True  # Validate on assignment
                allow_population_by_field_name = True  # Accept field names in input
                use_enum_values = True  # Use enum values in serialization
        
    except ImportError:
        print("pydantic not installed; pip install pydantic")


# ============================================================================
# 4. PATH & QUERY PARAMETERS (FASTAPI FOCUS)
# ============================================================================

def fastapi_path_query_params():
    """Path and query parameter handling in FastAPI"""
    try:
        from fastapi import FastAPI, Path, Query
        from typing import Optional
        
        app = FastAPI()
        
        # Path parameter with validation
        @app.get("/users/{user_id}")
        async def get_user(
            user_id: int = Path(..., gt=0, le=1000, description="User ID")
        ):
            return {"user_id": user_id}
        
        # Multiple path parameters
        @app.get("/users/{user_id}/posts/{post_id}")
        async def get_user_post(user_id: int, post_id: int):
            return {"user_id": user_id, "post_id": post_id}
        
        # Query parameters with Query object
        @app.get("/search")
        async def search(
            q: str = Query(..., min_length=1, max_length=100),
            skip: int = Query(0, ge=0),
            limit: int = Query(10, ge=1, le=100)
        ):
            return {"q": q, "skip": skip, "limit": limit}
        
        # List query parameters
        @app.get("/items")
        async def get_items(tags: Optional[List[str]] = Query(None)):
            # ?tags=python&tags=api -> tags = ["python", "api"]
            return {"tags": tags}
        
        # Enum parameters
        from enum import Enum
        
        class SortBy(str, Enum):
            name = "name"
            date = "date"
            rating = "rating"
        
        @app.get("/products")
        async def list_products(sort_by: SortBy = SortBy.name):
            return {"sort_by": sort_by}
        
    except ImportError:
        print("fastapi not installed")


# ============================================================================
# 5. REQUEST BODIES & FILE UPLOADS
# ============================================================================

def fastapi_request_bodies():
    """Request body handling in FastAPI"""
    try:
        from fastapi import FastAPI, File, UploadFile, Form
        from pydantic import BaseModel
        from typing import Optional, List
        
        app = FastAPI()
        
        # JSON body
        class Item(BaseModel):
            name: str
            price: float
            description: Optional[str] = None
        
        @app.post("/items")
        async def create_item(item: Item):
            return {"item": item.dict(), "total": item.price * 1.1}
        
        # File upload (single file)
        @app.post("/upload")
        async def upload_file(file: UploadFile = File(...)):
            content = await file.read()
            return {
                "filename": file.filename,
                "size": len(content),
                "content_type": file.content_type
            }
        
        # Multiple files
        @app.post("/upload-multiple")
        async def upload_files(files: List[UploadFile] = File(...)):
            return {
                "files": [f.filename for f in files]
            }
        
        # Form data (application/x-www-form-urlencoded)
        @app.post("/form")
        async def submit_form(
            username: str = Form(...),
            password: str = Form(...),
            remember: bool = Form(False)
        ):
            return {"username": username, "remember": remember}
        
        # Combined JSON + files
        @app.post("/upload-with-data")
        async def upload_with_data(
            item: Item,
            file: UploadFile = File(...)
        ):
            return {
                "item": item.dict(),
                "file": file.filename
            }
        
    except ImportError:
        print("fastapi not installed")


# ============================================================================
# 6. RESPONSE HANDLING
# ============================================================================

def fastapi_response_handling():
    """Controlling response format and status"""
    try:
        from fastapi import FastAPI, status
        from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
        from pydantic import BaseModel
        
        app = FastAPI()
        
        class Item(BaseModel):
            id: int
            name: str
        
        # Custom status code
        @app.post("/items", status_code=status.HTTP_201_CREATED)
        async def create_item(item: Item):
            return item
        
        # Custom response with headers
        @app.get("/custom")
        async def custom_response():
            return JSONResponse(
                status_code=200,
                content={"message": "Hello"},
                headers={"X-Custom": "value"}
            )
        
        # Redirect
        from fastapi.responses import RedirectResponse
        
        @app.get("/redirect")
        async def redirect():
            return RedirectResponse(url="/items")
        
        # File download
        @app.get("/download")
        async def download_file():
            return FileResponse(
                path="/path/to/file.txt",
                filename="downloaded.txt"
            )
        
        # Streaming response (for large files)
        async def generate():
            for i in range(10):
                yield f"data: {i}\n"
        
        @app.get("/stream")
        async def stream():
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Different response models based on conditions
        @app.get("/items/{item_id}")
        async def get_item(item_id: int, include_details: bool = False):
            item = {"id": item_id, "name": "Widget"}
            if include_details:
                item["description"] = "A useful widget"
            return item
        
    except ImportError:
        print("fastapi not installed")


# ============================================================================
# 7. ERROR HANDLING & EXCEPTIONS
# ============================================================================

def fastapi_error_handling():
    """Error handling and custom exceptions"""
    try:
        from fastapi import FastAPI, HTTPException, status
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, validator
        
        app = FastAPI()
        
        # Raise HTTPException
        @app.get("/items/{item_id}")
        async def get_item(item_id: int):
            if item_id == 999:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Item not found",
                    headers={"X-Error": "Not found"}
                )
            return {"item_id": item_id}
        
        # Custom exception class
        class ItemNotFound(Exception):
            def __init__(self, item_id: int):
                self.item_id = item_id
        
        @app.exception_handler(ItemNotFound)
        async def item_not_found_handler(request, exc):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": f"Item {exc.item_id} not found"}
            )
        
        @app.get("/products/{product_id}")
        async def get_product(product_id: int):
            if product_id < 0:
                raise ItemNotFound(product_id)
            return {"id": product_id}
        
        # Validation error handling
        from fastapi.exceptions import RequestValidationError
        
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request, exc):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": "Validation failed",
                    "errors": exc.errors()
                }
            )
        
        # Global error handling with middleware
        from fastapi.middleware.base import BaseHTTPMiddleware
        
        class ErrorHandlingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                try:
                    response = await call_next(request)
                    return response
                except Exception as exc:
                    logger.error(f"Unhandled error: {exc}")
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"detail": "Internal server error"}
                    )
        
        # app.add_middleware(ErrorHandlingMiddleware)
        
    except ImportError:
        print("fastapi not installed")


# ============================================================================
# 8. AUTHENTICATION & JWT
# ============================================================================

@dataclass
class TokenData:
    """Token payload structure"""
    user_id: int
    username: str
    exp: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24))


class JWTHandler:
    """JWT token creation and verification"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, user_id: int, username: str, expires_in_hours: int = 24) -> str:
        """Create JWT token"""
        exp = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": exp
        }
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("user_id")
            username = payload.get("username")
            exp = payload.get("exp")
            
            if not user_id or not username:
                return None
            
            return TokenData(
                user_id=user_id,
                username=username,
                exp=datetime.fromtimestamp(exp, tz=timezone.utc)
            )
        except jwt.InvalidTokenError:
            return None


def fastapi_jwt_authentication():
    """JWT authentication in FastAPI"""
    try:
        from fastapi import FastAPI, Depends, HTTPException, status
        from fastapi.security import HTTPBearer, HTTPAuthCredentials
        from pydantic import BaseModel
        
        app = FastAPI()
        
        # Setup
        jwt_handler = JWTHandler(secret_key="your-secret-key-change-this")
        security = HTTPBearer()
        
        # Models
        class LoginRequest(BaseModel):
            username: str
            password: str
        
        class LoginResponse(BaseModel):
            access_token: str
            token_type: str
        
        # Dependency: get current user from token
        async def get_current_user(credentials: HTTPAuthCredentials = Depends(security)):
            token = credentials.credentials
            token_data = jwt_handler.verify_token(token)
            
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            return token_data
        
        # Login endpoint
        @app.post("/login", response_model=LoginResponse)
        async def login(request: LoginRequest):
            # Validate credentials (simplified)
            if request.username == "admin" and request.password == "secret":
                token = jwt_handler.create_token(user_id=1, username="admin")
                return LoginResponse(access_token=token, token_type="bearer")
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
        
        # Protected endpoint
        @app.get("/profile")
        async def get_profile(user: TokenData = Depends(get_current_user)):
            return {"user_id": user.user_id, "username": user.username}
        
        # Get current user anywhere with Depends(get_current_user)
        
    except ImportError:
        print("fastapi not installed; pip install python-jose[cryptography]")


# ============================================================================
# 9. MIDDLEWARE & DECORATORS
# ============================================================================

def fastapi_middleware():
    """Middleware for request/response processing"""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.base import BaseHTTPMiddleware
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        import time
        
        app = FastAPI()
        
        # Custom middleware
        class TimingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                start = time.time()
                response = await call_next(request)
                duration = time.time() - start
                response.headers["X-Process-Time"] = str(duration)
                logger.info(f"{request.method} {request.url.path} took {duration:.3f}s")
                return response
        
        # Logging middleware
        class LoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                logger.info(f"Request: {request.method} {request.url.path}")
                response = await call_next(request)
                logger.info(f"Response: {response.status_code}")
                return response
        
        # Add middleware
        app.add_middleware(TimingMiddleware)
        app.add_middleware(LoggingMiddleware)
        
        # CORS middleware (for cross-origin requests)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "https://example.com"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "Authorization"]
        )
        
        # Trusted host middleware (security)
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["example.com", "www.example.com"]
        )
        
    except ImportError:
        print("fastapi not installed")


# ============================================================================
# 10. DATABASE INTEGRATION (SQLALCHEMY)
# ============================================================================

def sqlalchemy_models():
    """SQLAlchemy ORM models"""
    try:
        from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, create_engine
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import relationship
        from datetime import datetime
        
        Base = declarative_base()
        
        class User(Base):
            __tablename__ = "users"
            
            id = Column(Integer, primary_key=True)
            username = Column(String(50), unique=True, nullable=False)
            email = Column(String(100), unique=True, nullable=False)
            password_hash = Column(String(255), nullable=False)
            created_at = Column(DateTime, default=datetime.utcnow)
            
            # Relationship
            posts = relationship("Post", back_populates="author")
        
        class Post(Base):
            __tablename__ = "posts"
            
            id = Column(Integer, primary_key=True)
            title = Column(String(200), nullable=False)
            content = Column(String(5000), nullable=False)
            author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
            created_at = Column(DateTime, default=datetime.utcnow)
            
            # Relationship
            author = relationship("User", back_populates="posts")
        
        # Create database engine
        engine = create_engine("sqlite:///app.db")
        
        # Create tables
        # Base.metadata.create_all(engine)
        
    except ImportError:
        print("sqlalchemy not installed; pip install sqlalchemy")


def sqlalchemy_operations():
    """CRUD operations with SQLAlchemy"""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Setup
        engine = create_engine("sqlite:///app.db")
        Session = sessionmaker(bind=engine)
        
        # Using session
        session = Session()
        
        # Create
        # user = User(username="john", email="john@example.com", password_hash="...")
        # session.add(user)
        # session.commit()
        
        # Read
        # user = session.query(User).filter(User.username == "john").first()
        
        # Update
        # user.email = "newemail@example.com"
        # session.commit()
        
        # Delete
        # session.delete(user)
        # session.commit()
        
        # Query all
        # users = session.query(User).all()
        
        # Query with filter
        # active_users = session.query(User).filter(User.active == True).all()
        
        # Query with relationships
        # user_with_posts = session.query(User).filter(User.id == 1).first()
        # for post in user_with_posts.posts:
        #     print(post.title)
        
        session.close()
        
    except ImportError:
        print("sqlalchemy not installed")


# ============================================================================
# 11. TESTING APIs WITH PYTEST
# ============================================================================

def pytest_api_testing():
    """Testing APIs with pytest"""
    test_code = '''
import pytest
from fastapi.testclient import TestClient
from apis_part2 import app

client = TestClient(app)

def test_read_items():
    response = client.get("/items")
    assert response.status_code == 200
    assert response.json() == {"items": []}

def test_create_item():
    response = client.post("/items", json={"name": "Widget", "price": 9.99})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Widget"

def test_get_item_not_found():
    response = client.get("/items/999")
    assert response.status_code == 404

@pytest.fixture
def sample_item():
    # Setup
    item = {"name": "Test Item", "price": 5.99}
    yield item
    # Teardown (if needed)

def test_with_fixture(sample_item):
    response = client.post("/items", json=sample_item)
    assert response.status_code == 201
'''
    print("See test examples above (run with: pytest test_api.py)")


# ============================================================================
# 12. API DOCUMENTATION
# ============================================================================

def fastapi_auto_docs():
    """FastAPI automatic documentation"""
    try:
        from fastapi import FastAPI
        
        app = FastAPI(
            title="My API",
            description="A great API with automatic docs",
            version="1.0.0",
            docs_url="/docs",  # Swagger UI at /docs
            redoc_url="/redoc",  # ReDoc at /redoc
            openapi_url="/openapi.json"  # OpenAPI schema
        )
        
        # Documentation is automatic from:
        # - Route docstrings
        # - Type hints
        # - Pydantic models
        # - Response models
        
        @app.get("/items/{item_id}")
        async def get_item(item_id: int):
            """
            Get item by ID.
            
            - **item_id**: The ID of the item to retrieve
            """
            return {"item_id": item_id}
        
        # Visit http://localhost:8000/docs for Swagger UI
        # Visit http://localhost:8000/redoc for ReDoc
        
    except ImportError:
        print("fastapi not installed")


# ============================================================================
# 13. RATE LIMITING
# ============================================================================

class SimpleRateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # user_id -> [timestamps]
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        cutoff = now - 60  # Last minute
        
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Remove old requests
        self.requests[user_id] = [t for t in self.requests[user_id] if t > cutoff]
        
        # Check limit
        if len(self.requests[user_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[user_id].append(now)
        return True


def fastapi_rate_limiting():
    """Rate limiting in FastAPI"""
    try:
        from fastapi import FastAPI, Depends, HTTPException, status
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        
        app = FastAPI()
        limiter = Limiter(key_func=get_remote_address)
        
        @app.get("/api/search")
        @limiter.limit("10/minute")  # 10 requests per minute
        async def search(q: str):
            return {"query": q}
        
        # Manual rate limiting
        rate_limiter = SimpleRateLimiter(requests_per_minute=100)
        
        async def check_rate_limit(user_id: str):
            if not rate_limiter.is_allowed(user_id):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
        
        @app.get("/api/data")
        async def get_data(_: None = Depends(check_rate_limit)):
            return {"data": "value"}
        
    except ImportError:
        print("slowapi not installed; pip install slowapi")


# ============================================================================
# 14. CORS & SECURITY
# ============================================================================

def cors_and_security():
    """CORS configuration and security headers"""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        from fastapi.middleware.gzip import GZIPMiddleware
        
        app = FastAPI()
        
        # CORS (Cross-Origin Resource Sharing)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:8000",
                "https://example.com"
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
            max_age=600  # 10 minutes preflight cache
        )
        
        # Or allow all origins (development only!)
        # allow_origins=["*"]
        
        # Security: trusted host
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["example.com", "www.example.com"]
        )
        
        # Compression
        app.add_middleware(GZIPMiddleware, minimum_size=1000)
        
        # Security headers
        @app.middleware("http")
        async def add_security_headers(request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            return response
        
    except ImportError:
        print("fastapi not installed")


# ============================================================================
# 15. WEBSOCKETS
# ============================================================================

def fastapi_websockets():
    """Real-time communication with WebSockets"""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        
        app = FastAPI()
        
        class ConnectionManager:
            def __init__(self):
                self.active_connections: List[WebSocket] = []
            
            async def connect(self, websocket: WebSocket):
                await websocket.accept()
                self.active_connections.append(websocket)
            
            def disconnect(self, websocket: WebSocket):
                self.active_connections.remove(websocket)
            
            async def broadcast(self, message: str):
                for connection in self.active_connections:
                    await connection.send_text(message)
        
        manager = ConnectionManager()
        
        @app.websocket("/ws/chat")
        async def websocket_endpoint(websocket: WebSocket):
            await manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    await manager.broadcast(f"Message: {data}")
            except WebSocketDisconnect:
                manager.disconnect(websocket)
        
        # Client-side (JavaScript):
        # const ws = new WebSocket("ws://localhost:8000/ws/chat");
        # ws.onmessage = (e) => console.log(e.data);
        # ws.send("Hello");
        
    except ImportError:
        print("fastapi not installed")


# ============================================================================
# BEST PRACTICES & PATTERNS
# ============================================================================

"""
REST API BEST PRACTICES:

1. VERSIONING
   - /api/v1/users, /api/v2/users
   - Or use Accept header: Accept: application/vnd.myapi.v1+json
   - Allows gradual migration of clients

2. CONSISTENT ERROR RESPONSES
   {
       "error": {
           "code": "INVALID_INPUT",
           "message": "Invalid user data",
           "details": [
               {"field": "email", "message": "Invalid format"}
           ]
       }
   }

3. PAGINATION FOR LIST ENDPOINTS
   - Include: items, total, page, page_size, has_more
   - Support limit/offset or cursor-based pagination
   - Default reasonable limit (e.g., 20, max 100)

4. FILTERING & SORTING
   - ?sort=-created_at (desc), sort=name (asc)
   - ?filter[status]=active&filter[category]=tech
   - Document all filterable fields

5. SECURITY
   - HTTPS only in production
   - Validate all inputs
   - Rate limit to prevent abuse
   - Use strong authentication (JWT, OAuth2)
   - Log security events
   - Hide sensitive info in responses

6. CACHING
   - Set Cache-Control headers for safe endpoints
   - Use ETags for conditional requests
   - Let clients/proxies cache appropriately

7. LOGGING & MONITORING
   - Log all requests with ID for tracing
   - Log errors with full context
   - Monitor response times, error rates
   - Alert on anomalies

8. DOCUMENTATION
   - Keep docs up-to-date (automated if possible)
   - Include authentication requirements
   - Provide example requests/responses
   - Document all error codes

9. BACKWARDS COMPATIBILITY
   - Don't remove fields, deprecate them
   - Add new optional fields freely
   - Support old API versions for transition period

10. IDEMPOTENCY
    - POST operations should be idempotent
    - Use Idempotency-Key header for retries
    - Safe to retry without side effects
"""


if __name__ == "__main__":
    print("API Building Examples (FastAPI & Flask)")
    print("=" * 50)
    
    print("\nUncomment examples to run them")
    print("\nTo run FastAPI app:")
    print("  pip install fastapi uvicorn")
    print("  uvicorn apis_part2:app --reload")
    print("  Visit http://localhost:8000/docs for interactive docs")
