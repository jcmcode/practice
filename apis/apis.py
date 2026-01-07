"""
================================================================================
APIS IN PYTHON - COMPREHENSIVE CHEAT SHEET (PART 1: FUNDAMENTALS & CLIENTS)
================================================================================
A complete guide to HTTP, REST APIs, and API development in Python.
Covers HTTP basics, making requests, client patterns, and common libraries.

CONTENTS (PART 1):
1. HTTP Fundamentals: methods, status codes, headers
2. Making Requests: urllib, requests, httpx libraries
3. Request/Response Handling: headers, cookies, sessions
4. Error Handling: status codes, exceptions, retries
5. Authentication: basic auth, bearer tokens, API keys
6. URL Construction & Query Parameters: building URLs safely
7. JSON & Data Formats: request/response bodies
8. Session Management: persistent connections, cookies
9. Async HTTP: httpx async, concurrent requests
10. Common Patterns: SDKs, pagination, rate limiting
11. Testing HTTP Clients: mocking, fixtures, unit tests

CONTENTS (PART 2 - Building APIs):
12. Flask Basics: routing, request/response, blueprints
13. FastAPI Fundamentals: type hints, auto-docs, validation
14. Path & Query Parameters: URL routing, validation
15. Request Bodies: JSON, form data, file uploads
16. Response Handling: status codes, custom responses, streaming
17. Error Handling: exceptions, error responses, logging
18. Middleware & Decorators: request/response processing
19. Authentication & Authorization: JWT, OAuth2, role-based
20. Database Integration: SQLAlchemy, migrations, transactions
21. Testing APIs: pytest, fixtures, client testing
22. API Documentation: OpenAPI, Swagger, custom docs
23. Rate Limiting & Throttling: request limits, backoff
24. CORS & Security: headers, CSRF, content security
25. WebSockets: real-time communication
26. Deployment: containerization, cloud platforms
27. Monitoring & Logging: structured logs, metrics
28. Best Practices: versioning, consistency, error patterns

================================================================================
"""

import json
import time
import hashlib
import hmac
import base64
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
from functools import wraps
from urllib.parse import urlencode, parse_qs, urlparse, quote
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. HTTP FUNDAMENTALS
# ============================================================================

class HTTPMethod(Enum):
    """Standard HTTP methods and their semantics"""
    GET = "GET"           # Retrieve resource (safe, idempotent)
    POST = "POST"         # Create resource (not idempotent)
    PUT = "PUT"           # Replace entire resource (idempotent)
    PATCH = "PATCH"       # Partial update (may or may not be idempotent)
    DELETE = "DELETE"     # Remove resource (idempotent)
    HEAD = "HEAD"         # Like GET, but no response body (safe, idempotent)
    OPTIONS = "OPTIONS"   # Describe communication options (safe, idempotent)


class HTTPStatus(Enum):
    """Common HTTP status codes and meanings"""
    # 1xx: Informational
    CONTINUE = 100
    SWITCHING_PROTOCOLS = 101
    
    # 2xx: Success
    OK = 200                              # Request succeeded
    CREATED = 201                         # Resource created
    ACCEPTED = 202                        # Request accepted, processing async
    NO_CONTENT = 204                      # Success but no response body
    
    # 3xx: Redirection
    MOVED_PERMANENTLY = 301
    FOUND = 302
    SEE_OTHER = 303
    NOT_MODIFIED = 304
    TEMPORARY_REDIRECT = 307
    
    # 4xx: Client Error
    BAD_REQUEST = 400                     # Invalid request
    UNAUTHORIZED = 401                    # Auth required/failed
    FORBIDDEN = 403                       # Auth OK but not allowed
    NOT_FOUND = 404                       # Resource not found
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409                        # State conflict (e.g., duplicate)
    UNPROCESSABLE_ENTITY = 422            # Validation failed
    RATE_LIMITED = 429                    # Too many requests
    
    # 5xx: Server Error
    INTERNAL_SERVER_ERROR = 500           # Server error
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503             # Temporarily unavailable
    GATEWAY_TIMEOUT = 504


class HTTPHeaders:
    """Common HTTP headers and conventions"""
    # Request headers
    ACCEPT = "Accept"                           # Media types client accepts
    ACCEPT_ENCODING = "Accept-Encoding"         # gzip, deflate, br
    ACCEPT_LANGUAGE = "Accept-Language"
    AUTHORIZATION = "Authorization"             # Auth credentials
    CACHE_CONTROL = "Cache-Control"
    CONTENT_TYPE = "Content-Type"               # Body media type
    CONTENT_LENGTH = "Content-Length"
    USER_AGENT = "User-Agent"
    HOST = "Host"
    REFERER = "Referer"
    
    # Response headers
    SET_COOKIE = "Set-Cookie"
    LOCATION = "Location"                       # Redirect target
    ETAG = "ETag"                               # Resource version
    LAST_MODIFIED = "Last-Modified"
    CACHE_CONTROL_RESPONSE = "Cache-Control"
    
    # CORS headers
    ALLOW_ORIGIN = "Access-Control-Allow-Origin"
    ALLOW_METHODS = "Access-Control-Allow-Methods"
    ALLOW_HEADERS = "Access-Control-Allow-Headers"
    ALLOW_CREDENTIALS = "Access-Control-Allow-Credentials"


def http_concepts():
    """Key HTTP concepts"""
    concepts = {
        "idempotent": "Operation produces same result if repeated (GET, PUT, DELETE)",
        "safe": "Operation doesn't modify server state (GET, HEAD, OPTIONS)",
        "stateless": "Server doesn't retain client context between requests",
        "REST": "Representational State Transfer - architectural style using HTTP",
        "HATEOAS": "Hypermedia As The Engine Of Application State - links in responses",
        "content_negotiation": "Client and server agree on response format via Accept header",
        "conditional_requests": "Use ETag/Last-Modified to avoid re-fetching unchanged resources",
        "caching": "Server hints (Cache-Control) let clients/proxies cache responses",
    }
    return concepts


# ============================================================================
# 2. MAKING REQUESTS: URLLIB (STANDARD LIBRARY)
# ============================================================================

def urllib_basic_requests():
    """urllib - standard library HTTP client (basic, low-level)"""
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
    
    # Simple GET request
    try:
        with urlopen("https://api.github.com/users/octocat") as response:
            data = response.read()
            print(f"Status: {response.status}")
            print(f"Headers: {response.headers}")
            # Parse JSON if needed
            import json
            obj = json.loads(data)
    except HTTPError as e:
        print(f"HTTP Error: {e.code}")
    except URLError as e:
        print(f"URL Error: {e.reason}")
    
    # POST with custom headers
    url = "https://api.example.com/data"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "MyApp/1.0"
    }
    body = json.dumps({"key": "value"}).encode("utf-8")
    
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req) as response:
            result = json.loads(response.read())
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# 3. MAKING REQUESTS: REQUESTS LIBRARY (MOST POPULAR)
# ============================================================================

def requests_basic_operations():
    """requests - user-friendly HTTP library (recommended for sync)"""
    try:
        import requests
        
        # GET request
        response = requests.get("https://api.github.com/users/octocat")
        response.raise_for_status()  # Raise HTTPError for bad status
        data = response.json()        # Parse JSON automatically
        
        # With query parameters
        params = {"per_page": 10, "page": 1}
        response = requests.get(
            "https://api.github.com/repos/octocat/Hello-World/issues",
            params=params
        )
        
        # POST with JSON body
        payload = {"title": "Test", "body": "Description"}
        response = requests.post(
            "https://api.example.com/issues",
            json=payload,  # Automatically serializes to JSON
            headers={"Authorization": "Bearer token"}
        )
        
        # POST with form data
        form_data = {"username": "user", "password": "pass"}
        response = requests.post(
            "https://api.example.com/login",
            data=form_data  # Application/x-www-form-urlencoded
        )
        
        # Upload file
        with open("file.txt", "rb") as f:
            files = {"file": f}
            response = requests.post(
                "https://api.example.com/upload",
                files=files
            )
        
        # Multiple files
        files = {
            "image": ("pic.jpg", open("pic.jpg", "rb"), "image/jpeg"),
            "data": ("data.json", json.dumps({"key": "val"}), "application/json")
        }
        response = requests.post("https://api.example.com/upload", files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Headers: {response.headers}")
        print(f"Content: {response.text}")
        print(f"JSON: {response.json()}")
        
    except ImportError:
        print("requests not installed; pip install requests")


def requests_headers_and_auth():
    """Custom headers, authentication, cookies"""
    try:
        import requests
        
        # Custom headers
        headers = {
            "User-Agent": "MyApp/1.0",
            "Accept": "application/json",
            "X-Custom-Header": "value"
        }
        response = requests.get("https://api.example.com/data", headers=headers)
        
        # Basic authentication
        response = requests.get(
            "https://api.example.com/protected",
            auth=("username", "password")
        )
        
        # Bearer token (JWT, OAuth)
        headers = {"Authorization": "Bearer eyJhbGc..."}
        response = requests.get("https://api.example.com/data", headers=headers)
        
        # API Key (common pattern)
        headers = {"X-API-Key": "secret-key-12345"}
        response = requests.get("https://api.example.com/data", headers=headers)
        
        # Cookies
        cookies = {"session_id": "abc123", "user_id": "42"}
        response = requests.get("https://api.example.com/profile", cookies=cookies)
        
        # Access response cookies
        response = requests.get("https://api.example.com/login")
        session_cookies = response.cookies
        print(f"Cookies: {session_cookies}")
        
    except ImportError:
        print("requests not installed")


def requests_sessions():
    """Sessions for persistent connections and state"""
    try:
        import requests
        
        # Session maintains cookies, connection pooling, etc.
        session = requests.Session()
        
        # Set default headers
        session.headers.update({
            "User-Agent": "MyApp/1.0",
            "Authorization": "Bearer token"
        })
        
        # All requests in session reuse connection
        r1 = session.get("https://api.example.com/users")
        r2 = session.get("https://api.example.com/posts")  # Reuses connection
        
        # Cookies persist across requests
        session.post("https://api.example.com/login", data={"user": "admin", "pass": "secret"})
        # Subsequent requests have session cookies
        r3 = session.get("https://api.example.com/profile")
        
        # Mount custom adapter for timeout/retry
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1  # 1s, 2s, 4s delays
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Now requests auto-retry
        response = session.get("https://api.example.com/data")
        
        session.close()
        
    except ImportError:
        print("requests not installed")


def requests_streaming_and_timeout():
    """Streaming large responses, timeout handling"""
    try:
        import requests
        
        # Streaming large file (don't load all in memory)
        response = requests.get(
            "https://api.example.com/large-file.zip",
            stream=True,
            timeout=30  # Seconds
        )
        
        with open("downloaded.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        
        # Iterate lines (for streaming JSON lines)
        response = requests.get(
            "https://api.example.com/stream",
            stream=True
        )
        for line in response.iter_lines():
            if line:
                obj = json.loads(line)
                print(f"Received: {obj}")
        
        # Timeout configurations
        response = requests.get(
            "https://api.example.com/data",
            timeout=5  # Total timeout: connect + read
        )
        
        response = requests.get(
            "https://api.example.com/data",
            timeout=(3, 10)  # Connect timeout=3s, read timeout=10s
        )
        
    except ImportError:
        print("requests not installed")


# ============================================================================
# 4. MAKING REQUESTS: HTTPX (MODERN, ASYNC-READY)
# ============================================================================

def httpx_basic_requests():
    """httpx - modern alternative with async support (pip install httpx)"""
    try:
        import httpx
        
        # Simple GET
        client = httpx.Client()
        response = client.get("https://api.github.com/users/octocat")
        print(response.status_code)
        print(response.json())
        
        # POST with JSON
        response = client.post(
            "https://api.example.com/data",
            json={"key": "value"},
            headers={"Authorization": "Bearer token"}
        )
        
        # Timeout
        response = client.get(
            "https://api.example.com/data",
            timeout=10
        )
        
        # Request limits & configuration
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        client = httpx.Client(limits=limits, timeout=30)
        
        # Streaming
        with client.stream("GET", "https://api.example.com/stream") as response:
            for line in response.iter_lines():
                print(line)
        
        client.close()
        
    except ImportError:
        print("httpx not installed; pip install httpx")


# ============================================================================
# 5. ERROR HANDLING & RETRIES
# ============================================================================

def error_handling_and_retries():
    """Handling errors, retries, backoff strategies"""
    try:
        import requests
        
        # Basic error handling
        try:
            response = requests.get("https://api.example.com/data", timeout=5)
            response.raise_for_status()  # Raise HTTPError for 4xx/5xx
            data = response.json()
        except requests.exceptions.Timeout:
            print("Request timed out")
        except requests.exceptions.ConnectionError:
            print("Connection failed")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
        
        # Manual retry with exponential backoff
        def retry_with_backoff(url, max_retries=3, base_delay=1):
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    return response
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # exponential backoff
                        print(f"Retry {attempt + 1} after {delay}s")
                        time.sleep(delay)
                    else:
                        raise
        
        response = retry_with_backoff("https://api.example.com/data")
        
    except ImportError:
        print("requests not installed")


def custom_retry_decorator():
    """Decorator for automatic retries"""
    
    def retry(max_attempts=3, delay=1, backoff_factor=2):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            wait_time = delay * (backoff_factor ** attempt)
                            print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s")
                            time.sleep(wait_time)
                        else:
                            raise
            return wrapper
        return decorator
    
    @retry(max_attempts=3, delay=1)
    def fetch_data(url):
        import requests
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    
    # Usage: fetch_data("https://api.example.com/data")


# ============================================================================
# 6. AUTHENTICATION PATTERNS
# ============================================================================

class AuthStrategy:
    """Base class for authentication strategies"""
    
    def apply(self, headers: Dict[str, str]) -> Dict[str, str]:
        raise NotImplementedError


class BasicAuth(AuthStrategy):
    """HTTP Basic Authentication (base64 encoded username:password)"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
    
    def apply(self, headers: Dict[str, str]) -> Dict[str, str]:
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"
        return headers


class BearerToken(AuthStrategy):
    """Bearer Token Authentication (OAuth2, JWT)"""
    
    def __init__(self, token: str):
        self.token = token
    
    def apply(self, headers: Dict[str, str]) -> Dict[str, str]:
        headers["Authorization"] = f"Bearer {self.token}"
        return headers


class APIKeyAuth(AuthStrategy):
    """API Key Authentication"""
    
    def __init__(self, key: str, header_name: str = "X-API-Key"):
        self.key = key
        self.header_name = header_name
    
    def apply(self, headers: Dict[str, str]) -> Dict[str, str]:
        headers[self.header_name] = self.key
        return headers


class HMACAuth(AuthStrategy):
    """HMAC Signature Authentication (e.g., AWS, Stripe)"""
    
    def __init__(self, key_id: str, secret: str):
        self.key_id = key_id
        self.secret = secret
    
    def sign_request(self, method: str, path: str, body: str = "") -> str:
        """Create HMAC-SHA256 signature"""
        message = f"{method}\n{path}\n{body}"
        signature = hmac.new(
            self.secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def apply(self, headers: Dict[str, str]) -> Dict[str, str]:
        headers["X-Signature"] = self.sign_request("POST", "/api/data")
        headers["X-Key-ID"] = self.key_id
        return headers


def auth_examples():
    """Using different auth strategies"""
    try:
        import requests
        
        # Basic auth
        auth = BasicAuth("user", "pass")
        headers = {}
        auth.apply(headers)
        # response = requests.get("https://api.example.com/data", headers=headers)
        
        # Bearer token (JWT)
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        auth = BearerToken(jwt_token)
        headers = {}
        auth.apply(headers)
        
        # API key
        auth = APIKeyAuth("sk_live_123456", header_name="X-API-Key")
        headers = {}
        auth.apply(headers)
        
        # HMAC signature
        auth = HMACAuth("key_id_123", "secret_key")
        headers = {}
        auth.apply(headers)
        
    except ImportError:
        print("requests not installed")


# ============================================================================
# 7. URL CONSTRUCTION & QUERY PARAMETERS
# ============================================================================

def url_building_and_params():
    """Safely construct URLs with parameters"""
    
    # Using urlencode
    params = {
        "search": "python api",
        "limit": 10,
        "offset": 0,
        "sort": "date"
    }
    query_string = urlencode(params)
    print(f"Query string: {query_string}")
    # Output: search=python+api&limit=10&offset=0&sort=date
    
    # Special characters are automatically escaped
    params = {
        "q": "hello world",  # Spaces -> %20 or +
        "name": "John & Jane",  # & -> %26
        "url": "https://example.com?foo=bar"  # Special chars escaped
    }
    query_string = urlencode(params)
    print(f"Safe query string: {query_string}")
    
    # Parse query string
    query_string = "search=python&limit=10&offset=0"
    parsed = parse_qs(query_string)
    print(f"Parsed: {parsed}")
    # Output: {'search': ['python'], 'limit': ['10'], 'offset': ['0']}
    
    # Parse URL
    url = "https://api.example.com/users?id=123&filter=active"
    parsed_url = urlparse(url)
    print(f"Scheme: {parsed_url.scheme}")
    print(f"Netloc: {parsed_url.netloc}")
    print(f"Path: {parsed_url.path}")
    print(f"Query: {parsed_url.query}")
    
    # Safe path encoding (prevent directory traversal)
    user_input = "my file.txt"
    safe_path = quote(user_input, safe="")
    print(f"Safe path: {safe_path}")  # my%20file.txt


# ============================================================================
# 8. REQUEST/RESPONSE DATA MODELS
# ============================================================================

@dataclass
class HttpRequest:
    """Type-safe HTTP request representation"""
    method: str
    url: str
    headers: Dict[str, str] = None
    query_params: Dict[str, Any] = None
    body: Optional[str] = None
    auth: Optional[AuthStrategy] = None
    timeout: Optional[float] = 30
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.query_params is None:
            self.query_params = {}
        
        # Apply auth to headers
        if self.auth:
            self.auth.apply(self.headers)


@dataclass
class HttpResponse:
    """Type-safe HTTP response representation"""
    status_code: int
    headers: Dict[str, str]
    body: str
    request: HttpRequest = None
    elapsed: float = None  # Time in seconds
    
    def json(self) -> Dict[str, Any]:
        """Parse response body as JSON"""
        return json.loads(self.body)
    
    def is_success(self) -> bool:
        """Check if response is 2xx"""
        return 200 <= self.status_code < 300
    
    def is_error(self) -> bool:
        """Check if response is 4xx or 5xx"""
        return self.status_code >= 400


# ============================================================================
# 9. CLIENT PATTERNS & UTILITIES
# ============================================================================

class APIClient:
    """Base HTTP API client with common patterns"""
    
    def __init__(self, base_url: str, auth: Optional[AuthStrategy] = None, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout
        self.headers = {"User-Agent": "APIClient/1.0"}
        
        try:
            import requests
            self.requests = requests
            self.session = requests.Session()
        except ImportError:
            self.requests = None
            self.session = None
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint"""
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> HttpResponse:
        """Make HTTP request with standard handling"""
        if not self.session:
            raise RuntimeError("requests not installed")
        
        url = self._build_url(endpoint)
        headers = {**self.headers, **kwargs.pop("headers", {})}
        
        # Apply auth
        if self.auth:
            self.auth.apply(headers)
        
        kwargs.setdefault("timeout", self.timeout)
        
        try:
            response = self.session.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            
            return HttpResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response.text
            )
        except self.requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
    
    def get(self, endpoint: str, params: Dict = None, **kwargs) -> HttpResponse:
        """GET request"""
        return self._make_request("GET", endpoint, params=params, **kwargs)
    
    def post(self, endpoint: str, json: Dict = None, **kwargs) -> HttpResponse:
        """POST request"""
        return self._make_request("POST", endpoint, json=json, **kwargs)
    
    def put(self, endpoint: str, json: Dict = None, **kwargs) -> HttpResponse:
        """PUT request (full replacement)"""
        return self._make_request("PUT", endpoint, json=json, **kwargs)
    
    def patch(self, endpoint: str, json: Dict = None, **kwargs) -> HttpResponse:
        """PATCH request (partial update)"""
        return self._make_request("PATCH", endpoint, json=json, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> HttpResponse:
        """DELETE request"""
        return self._make_request("DELETE", endpoint, **kwargs)
    
    def close(self):
        """Close session"""
        if self.session:
            self.session.close()


# ============================================================================
# 10. PAGINATION PATTERNS
# ============================================================================

class Paginator:
    """Handle paginated API responses"""
    
    def __init__(self, client: APIClient, endpoint: str, page_size: int = 10):
        self.client = client
        self.endpoint = endpoint
        self.page_size = page_size
    
    def fetch_all(self) -> List[Dict]:
        """Fetch all pages"""
        all_items = []
        page = 1
        
        while True:
            response = self.client.get(
                self.endpoint,
                params={"page": page, "per_page": self.page_size}
            )
            
            if not response.is_success():
                raise RuntimeError(f"API error: {response.status_code}")
            
            items = response.json()
            if not items:
                break
            
            all_items.extend(items)
            page += 1
        
        return all_items
    
    def fetch_paginated(self):
        """Generator for lazy fetching"""
        page = 1
        
        while True:
            response = self.client.get(
                self.endpoint,
                params={"page": page, "per_page": self.page_size}
            )
            
            if not response.is_success():
                break
            
            items = response.json()
            if not items:
                break
            
            for item in items:
                yield item
            
            page += 1


# ============================================================================
# 11. RATE LIMITING PATTERNS
# ============================================================================

class RateLimiter:
    """Simple token bucket rate limiter"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def wait_if_needed(self):
        """Block if rate limit exceeded"""
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Remove old requests outside window
        self.requests = [t for t in self.requests if t > cutoff]
        
        if len(self.requests) >= self.max_requests:
            # Wait until oldest request leaves window
            sleep_time = self.requests[0] - cutoff
            print(f"Rate limit reached, waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self.requests = []
        
        self.requests.append(now)


def rate_limited_requests():
    """Making requests with rate limiting"""
    
    # 10 requests per minute
    limiter = RateLimiter(max_requests=10, window_seconds=60)
    
    # urls = [...]
    # for url in urls:
    #     limiter.wait_if_needed()
    #     response = requests.get(url)


# ============================================================================
# 12. TESTING HTTP CLIENTS
# ============================================================================

def testing_http_clients():
    """Examples of testing HTTP clients"""
    
    # Using responses library to mock HTTP
    try:
        import responses
        
        @responses.activate
        def test_github_api():
            # Mock the endpoint
            responses.add(
                responses.GET,
                "https://api.github.com/users/octocat",
                json={"login": "octocat", "id": 1, "public_repos": 2},
                status=200
            )
            
            # Test code
            client = APIClient(base_url="https://api.github.com")
            response = client.get("/users/octocat")
            
            assert response.status_code == 200
            data = response.json()
            assert data["login"] == "octocat"
        
        # test_github_api()
    except ImportError:
        print("responses not installed; pip install responses")


# ============================================================================
# REST PRINCIPLES SUMMARY
# ============================================================================

"""
REST (Representational State Transfer) Principles:

1. CLIENT-SERVER ARCHITECTURE
   - Client: makes requests
   - Server: processes and responds
   - Loosely coupled, independent scaling

2. STATELESSNESS
   - Each request contains all info needed
   - Server doesn't store client context
   - Simplifies scaling and caching

3. RESOURCE-ORIENTED
   - Resources identified by URIs: /users/123, /posts/456/comments
   - Not action-oriented: avoid /get_user, /create_post
   - Use HTTP methods to express actions

4. STANDARD METHODS
   - GET: retrieve (safe, idempotent)
   - POST: create (not idempotent)
   - PUT: full replace (idempotent)
   - PATCH: partial update
   - DELETE: remove (idempotent)

5. CONSISTENT STATUS CODES
   - 2xx: Success
   - 3xx: Redirect
   - 4xx: Client error (bad request, not found, forbidden)
   - 5xx: Server error

6. CONTENT NEGOTIATION
   - Accept header: client specifies desired format
   - Content-Type: server specifies response format
   - Usually JSON in modern APIs

7. CACHING
   - Use Cache-Control, ETag, Last-Modified
   - Allow clients to avoid redundant requests

8. HATEOAS (Advanced)
   - Include links in responses to guide clients
   - Self-documenting API navigation
   - Often skipped in practice for simplicity
"""


# ============================================================================
# EXAMPLE: COMPLETE API CLIENT
# ============================================================================

@dataclass
class User:
    id: int
    username: str
    email: str


class GitHubAPIClient(APIClient):
    """Example: GitHub API client"""
    
    def __init__(self, token: str):
        auth = BearerToken(token)
        super().__init__(base_url="https://api.github.com", auth=auth)
    
    def get_user(self, username: str) -> Dict:
        """Get user info"""
        response = self.get(f"/users/{username}")
        return response.json()
    
    def get_repos(self, username: str) -> List[Dict]:
        """Get user repos"""
        response = self.get(f"/users/{username}/repos")
        return response.json()
    
    def get_issues(self, owner: str, repo: str) -> List[Dict]:
        """Get repository issues"""
        response = self.get(f"/repos/{owner}/{repo}/issues")
        return response.json()


# ============================================================================
# UTILITIES & HELPERS
# ============================================================================

def pretty_print_response(response: HttpResponse):
    """Pretty print API response"""
    print(f"Status: {response.status_code}")
    print(f"Headers:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    print(f"Body:")
    try:
        pretty_json = json.dumps(response.json(), indent=2)
        print(pretty_json)
    except:
        print(response.body)


if __name__ == "__main__":
    print("HTTP & API Client Examples")
    print("=" * 50)
    
    # http_concepts()
    # urllib_basic_requests()
    # requests_basic_operations()
    # requests_headers_and_auth()
    # requests_sessions()
    # error_handling_and_retries()
    # auth_examples()
    # url_building_and_params()
    
    print("\nUncomment examples to run them")
    print("See PART 2 for building APIs with Flask/FastAPI")
