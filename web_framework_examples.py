#!/usr/bin/env python3
"""
Web Framework Examples
======================

This module demonstrates web development with Python frameworks:
- Flask (microframework)
- FastAPI (async framework)
- Basic routing, templates, and APIs
- Database integration
- Authentication and middleware
- Deployment considerations
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import os

# Try to import web frameworks
try:
    from flask import Flask, request, jsonify, render_template_string, make_response
    from flask_sqlalchemy import SQLAlchemy
    from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Note: Flask not installed. Run: pip install flask flask-sqlalchemy flask-login")

try:
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field, validator
    from sqlalchemy import create_engine, Column, Integer, String, DateTime
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Note: FastAPI not installed. Run: pip install fastapi uvicorn sqlalchemy pydantic")


# ---------- Flask Examples ----------

if FLASK_AVAILABLE:
    # Flask app setup
    flask_app = Flask(__name__)
    flask_app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flask_example.db'
    flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db = SQLAlchemy(flask_app)
    login_manager = LoginManager()
    login_manager.init_app(flask_app)
    login_manager.login_view = 'login'
    
    
    # Models
    class User(UserMixin, db.Model):
        """User model for Flask app."""
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(128))
        created_at = db.Column(db.DateTime, default=datetime.now)
        
        posts = db.relationship('Post', backref='author', lazy=True)
        
        def __repr__(self):
            return f'<User {self.username}>'
    
    
    class Post(db.Model):
        """Blog post model."""
        id = db.Column(db.Integer, primary_key=True)
        title = db.Column(db.String(200), nullable=False)
        content = db.Column(db.Text, nullable=False)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
        created_at = db.Column(db.DateTime, default=datetime.now)
        published = db.Column(db.Boolean, default=False)
        
        def to_dict(self):
            """Convert to dictionary."""
            return {
                'id': self.id,
                'title': self.title,
                'content': self.content,
                'user_id': self.user_id,
                'author': self.author.username if self.author else None,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'published': self.published
            }
    
    
    @login_manager.user_loader
    def load_user(user_id):
        """Load user by ID."""
        return User.query.get(int(user_id))
    
    
    # Routes
    @flask_app.route('/')
    def home():
        """Home page."""
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flask Example</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .nav { background: #f4f4f4; padding: 10px; margin-bottom: 20px; }
                .nav a { margin-right: 15px; text-decoration: none; }
                .post { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; }
            </style>
        </head>
        <body>
            <div class="nav">
                <a href="/">Home</a>
                <a href="/api/posts">API</a>
                <a href="/login">Login</a>
                <a href="/register">Register</a>
                <a href="/create">Create Post</a>
            </div>
            <h1>Welcome to Flask Blog</h1>
            <p>A simple blog built with Flask</p>
            <h2>Features:</h2>
            <ul>
                <li>User authentication</li>
                <li>CRUD operations</li>
                <li>RESTful API</li>
                <li>SQLite database</li>
                <li>Templates with Jinja2</li>
            </ul>
            <h2>Recent Posts:</h2>
            {% for post in posts %}
            <div class="post">
                <h3>{{ post.title }}</h3>
                <p>{{ post.content[:100] }}...</p>
                <small>By {{ post.author }} on {{ post.created_at }}</small>
            </div>
            {% endfor %}
        </body>
        </html>
        """, posts=Post.query.filter_by(published=True).order_by(Post.created_at.desc()).limit(5).all())
    
    
    @flask_app.route('/api/posts')
    def api_posts():
        """API endpoint to get all posts."""
        posts = Post.query.filter_by(published=True).all()
        return jsonify([post.to_dict() for post in posts])
    
    
    @flask_app.route('/api/posts/<int:post_id>')
    def api_post(post_id):
        """API endpoint to get a specific post."""
        post = Post.query.get_or_404(post_id)
        return jsonify(post.to_dict())
    
    
    @flask_app.route('/api/posts', methods=['POST'])
    @login_required
    def api_create_post():
        """API endpoint to create a new post."""
        data = request.get_json()
        
        if not data or 'title' not in data or 'content' not in data:
            return jsonify({'error': 'Missing title or content'}), 400
        
        post = Post(
            title=data['title'],
            content=data['content'],
            user_id=current_user.id,
            published=data.get('published', False)
        )
        
        db.session.add(post)
        db.session.commit()
        
        return jsonify(post.to_dict()), 201
    
    
    @flask_app.route('/login', methods=['GET', 'POST'])
    def login():
        """Login page."""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            # Simplified authentication (in production, use password hashing)
            user = User.query.filter_by(username=username).first()
            if user and user.password_hash == password:  # In real app, use proper hashing
                login_user(user)
                return 'Logged in successfully'
            
            return 'Invalid credentials'
        
        return render_template_string("""
        <h1>Login</h1>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required><br>
            <input type="password" name="password" placeholder="Password" required><br>
            <button type="submit">Login</button>
        </form>
        """)
    
    
    @flask_app.route('/logout')
    @login_required
    def logout():
        """Logout endpoint."""
        logout_user()
        return 'Logged out'
    
    
    @flask_app.route('/create', methods=['GET', 'POST'])
    @login_required
    def create_post():
        """Create a new post."""
        if request.method == 'POST':
            title = request.form.get('title')
            content = request.form.get('content')
            published = 'published' in request.form
            
            post = Post(
                title=title,
                content=content,
                user_id=current_user.id,
                published=published
            )
            
            db.session.add(post)
            db.session.commit()
            
            return f'Post "{title}" created successfully'
        
        return render_template_string("""
        <h1>Create Post</h1>
        <form method="POST">
            <input type="text" name="title" placeholder="Title" required><br>
            <textarea name="content" placeholder="Content" rows="10" cols="50" required></textarea><br>
            <label>
                <input type="checkbox" name="published"> Publish immediately
            </label><br>
            <button type="submit">Create Post</button>
        </form>
        """)
    
    
    # Error handlers
    @flask_app.errorhandler(404)
    def not_found(error):
        """404 error handler."""
        return jsonify({'error': 'Not found'}), 404
    
    
    @flask_app.errorhandler(500)
    def internal_error(error):
        """500 error handler."""
        return jsonify({'error': 'Internal server error'}), 500
    
    
    def setup_flask_database():
        """Set up Flask database with sample data."""
        with flask_app.app_context():
            # Create all tables
            db.create_all()
            
            # Add sample user if none exists
            if User.query.count() == 0:
                user = User(
                    username='admin',
                    email='admin@example.com',
                    password_hash='password'  # In production, use proper hashing
                )
                db.session.add(user)
                db.session.commit()
                
                # Add sample posts
                posts = [
                    Post(title='Welcome to Flask Blog', content='This is the first post.', user_id=user.id, published=True),
                    Post(title='Python Web Development', content='Learn how to build web apps with Flask.', user_id=user.id, published=True),
                    Post(title='Database Integration', content='Using SQLAlchemy with Flask.', user_id=user.id, published=True),
                    Post(title='Authentication Guide', content='Implementing user auth in Flask.', user_id=user.id, published=False),
                ]
                
                for post in posts:
                    db.session.add(post)
                
                db.session.commit()
                print("Created sample data for Flask app")


# ---------- FastAPI Examples ----------

if FASTAPI_AVAILABLE:
    # FastAPI app setup
    fastapi_app = FastAPI(
        title="FastAPI Example",
        description="A modern, fast web framework for Python",
        version="1.0.0"
    )
    
    # Database setup
    SQLALCHEMY_DATABASE_URL = "sqlite:///./fastapi_example.db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    
    
    # Models
    class FastAPIUser(Base):
        """User model for FastAPI."""
        __tablename__ = "fastapi_users"
        
        id = Column(Integer, primary_key=True, index=True)
        username = Column(String(50), unique=True, index=True, nullable=False)
        email = Column(String(100), unique=True, index=True, nullable=False)
        full_name = Column(String(100))
        disabled = Column(Integer, default=0)  # 0=False, 1=True
        created_at = Column(DateTime, default=datetime.now)
    
    
    class FastAPIPost(Base):
        """Post model for FastAPI."""
        __tablename__ = "fastapi_posts"
        
        id = Column(Integer, primary_key=True, index=True)
        title = Column(String(200), nullable=False)
        content = Column(String, nullable=False)
        user_id = Column(Integer, nullable=False)
        published = Column(Integer, default=0)  # 0=False, 1=True
        created_at = Column(DateTime, default=datetime.now)
    
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    
    # Pydantic models (request/response schemas)
    class UserBase(BaseModel):
        """Base user schema."""
        username: str
        email: str
        full_name: Optional[str] = None
        disabled: Optional[bool] = False
    
    
    class UserCreate(UserBase):
        """User creation schema."""
        password: str
    
    
    class UserResponse(UserBase):
        """User response schema."""
        id: int
        created_at: datetime
        
        class Config:
            from_attributes = True
    
    
    class PostBase(BaseModel):
        """Base post schema."""
        title: str
        content: str
        published: bool = False
    
    
    class PostCreate(PostBase):
        """Post creation schema."""
        pass
    
    
    class PostResponse(PostBase):
        """Post response schema."""
        id: int
        user_id: int
        created_at: datetime
        
        class Config:
            from_attributes = True
    
    
    # Dependency
    def get_db():
        """Get database session."""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    
    # Security
    security = HTTPBearer()
    
    
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify authentication token (simplified)."""
        # In production, verify JWT or other token
        token = credentials.credentials
        
        # Simple demo: token is username
        db = next(get_db())
        user = db.query(FastAPIUser).filter(FastAPIUser.username == token).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
    
    
    # Routes
    @fastapi_app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Welcome to FastAPI Example",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    
    
    @fastapi_app.get("/users/", response_model=List[UserResponse])
    async def get_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
        """Get all users with pagination."""
        users = db.query(FastAPIUser).offset(skip).limit(limit).all()
        return users
    
    
    @fastapi_app.get("/users/{user_id}", response_model=UserResponse)
    async def get_user(user_id: int, db: Session = Depends(get_db)):
        """Get a specific user."""
        user = db.query(FastAPIUser).filter(FastAPIUser.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    
    
    @fastapi_app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    async def create_user(user: UserCreate, db: Session = Depends(get_db)):
        """Create a new user."""
        # Check if username or email already exists
        existing_user = db.query(FastAPIUser).filter(
            (FastAPIUser.username == user.username) | (FastAPIUser.email == user.email)
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        # Create new user (password would be hashed in production)
        db_user = FastAPIUser(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            disabled=1 if user.disabled else 0
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return db_user
    
    
    @fastapi_app.get("/posts/", response_model=List[PostResponse])
    async def get_posts(
        skip: int = 0,
        limit: int = 10,
        published: Optional[bool] = None,
        db: Session = Depends(get_db)
    ):
        """Get all posts with optional filtering."""
        query = db.query(FastAPIPost)
        
        if published is not None:
            query = query.filter(FastAPIPost.published == (1 if published else 0))
        
        posts = query.offset(skip).limit(limit).all()
        return posts
    
    
    @fastapi_app.post("/posts/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
    async def create_post(post: PostCreate, current_user: FastAPIUser = Depends(verify_token)):
        """Create a new post (authenticated)."""
        db = next(get_db())
        
        db_post = FastAPIPost(
            title=post.title,
            content=post.content,
            user_id=current_user.id,
            published=1 if post.published else 0
        )
        
        db.add(db_post)
        db.commit()
        db.refresh(db_post)
        
        return db_post
    
    
    @fastapi_app.get("/posts/{post_id}", response_model=PostResponse)
    async def get_post(post_id: int, db: Session = Depends(get_db)):
        """Get a specific post."""
        post = db.query(FastAPIPost).filter(FastAPIPost.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        return post
    
    
    @fastapi_app.put("/posts/{post_id}", response_model=PostResponse)
    async def update_post(
        post_id: int,
        post_update: PostCreate,
        current_user: FastAPIUser = Depends(verify_token),
        db: Session = Depends(get_db)
    ):
        """Update a post (authenticated)."""
        post = db.query(FastAPIPost).filter(FastAPIPost.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Check ownership
        if post.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to update this post")
        
        # Update post
        post.title = post_update.title
        post.content = post_update.content
        post.published = 1 if post_update.published else 0
        
        db.commit()
        db.refresh(post)
        
        return post
    
    
    @fastapi_app.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_post(
        post_id: int,
        current_user: FastAPIUser = Depends(verify_token),
        db: Session = Depends(get_db)
    ):
        """Delete a post (authenticated)."""
        post = db.query(FastAPIPost).filter(FastAPIPost.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Check ownership
        if post.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this post")
        
        db.delete(post)
        db.commit()
        
        return None
    
    
    def setup_fastapi_database():
        """Set up FastAPI database with sample data."""
        db = SessionLocal()
        
        try:
            # Add sample user if none exists
            if db.query(FastAPIUser).count() == 0:
                user = FastAPIUser(
                    username="demo_user",
                    email="demo@example.com",
                    full_name="Demo User",
                    disabled=0
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                
                # Add sample posts
                posts = [
                    FastAPIPost(
                        title="Welcome to FastAPI",
                        content="This is a sample post created with FastAPI.",
                        user_id=user.id,
                        published=1
                    ),
                    FastAPIPost(
                        title="Python Async Web Framework",
                        content="FastAPI makes it easy to build async web applications.",
                        user_id=user.id,
                        published=1
                    ),
                    FastAPIPost(
                        title="API Documentation",
                        content="FastAPI automatically generates OpenAPI documentation.",
                        user_id=user.id,
                        published=0
                    ),
                ]
                
                for post in posts:
                    db.add(post)
                
                db.commit()
                print("Created sample data for FastAPI app")
        
        finally:
            db.close()


# ---------- Run Examples ----------

def run_flask_example():
    """Run Flask example."""
    if not FLASK_AVAILABLE:
        print("\nFlask not installed. Install with: pip install flask flask-sqlalchemy flask-login")
        return
    
    print("\n" + "="*60)
    print("Flask Web Framework Example")
    print("="*60)
    
    # Setup database
    setup_flask_database()
    
    print("\nFlask Application Created!")
    print("Key Endpoints:")
    print("  - Home: http://localhost:5000/")
    print("  - API Posts: http://localhost:5000/api/posts")
    print("  - Login: http://localhost:5000/login")
    print("  - Create Post: http://localhost:5000/create")
    
    print("\nTo run the Flask app:")
    print("  export FLASK_APP=web_framework_examples.py")
    print("  export FLASK_ENV=development")
    print("  flask run")
    
    print("\nDatabase file: flask_example.db")
    print("Sample user: admin / password")


def run_fastapi_example():
    """Run FastAPI example."""
    if not FASTAPI_AVAILABLE:
        print("\nFastAPI not installed. Install with: pip install fastapi uvicorn sqlalchemy pydantic")
        return
    
    print("\n" + "="*60)
    print("FastAPI Web Framework Example")
    print("="*60)
    
    # Setup database
    setup_fastapi_database()
    
    print("\nFastAPI Application Created!")
    print("Key Endpoints:")
    print("  - Root: http://localhost:8000/")
    print("  - Docs: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("  - Users: http://localhost:8000/users/")
    print("  - Posts: http://localhost:8000/posts/")
    
    print("\nTo run the FastAPI app:")
    print("  uvicorn web_framework_examples:fastapi_app --reload")
    
    print("\nAuthentication:")
    print("  Use 'demo_user' as Bearer token for protected endpoints")
    print("  Example: curl -H 'Authorization: Bearer demo_user' http://localhost:8000/posts/")
    
    print("\nDatabase file: fastapi_example.db")
    print("Sample user: demo_user")


def create_requirements_file():
    """Create requirements file for web frameworks."""
    requirements = """# Web Framework Dependencies
# Flask
flask>=2.3.0
flask-sqlalchemy>=3.0.0
flask-login>=0.6.0
python-dotenv>=1.0.0

# FastAPI
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
sqlalchemy>=2.0.0
pydantic>=2.0.0

# Common
jinja2>=3.1.0
"""
    
    with open("requirements-web.txt", "w") as f:
        f.write(requirements)
    
    print("\nCreated requirements-web.txt file")


def create_docker_compose():
    """Create Docker Compose file for web apps."""
    docker_compose = """version: '3.8'

services:
  flask-app:
    build:
      context: .
      dockerfile: Dockerfile.flask
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
      - DATABASE_URL=sqlite:///app.db
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  fastapi-app:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///app.db
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=app_db
      - POSTGRES_USER=app_user
      - POSTGRES_PASSWORD=app_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    dockerfile_flask = """FROM python:3.11-slim

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . .

ENV FLASK_APP=web_framework_examples.py
ENV FLASK_ENV=development
ENV DATABASE_URL=sqlite:///data/app.db

CMD ["flask", "run", "--host=0.0.0.0"]
"""
    
    with open("Dockerfile.flask", "w") as f:
        f.write(dockerfile_flask)
    
    dockerfile_fastapi = """FROM python:3.11-slim

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . .

ENV DATABASE_URL=sqlite:///data/app.db

CMD ["uvicorn", "web_framework_examples:fastapi_app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open("Dockerfile.fastapi", "w") as f:
        f.write(dockerfile_fastapi)
    
    print("Created Docker Compose and Dockerfiles for web apps")


# ---------- Main Execution ----------

def main():
    """Run all web framework examples."""
    print("Web Framework Examples")
    print("="*60)
    
    # Run Flask example
    run_flask_example()
    
    # Run FastAPI example
    run_fastapi_example()
    
    # Create supporting files
    create_requirements_file()
    create_docker_compose()
    
    print("\n" + "="*60)
    print("Web Framework Examples Complete!")
    print("="*60)
    
    # Create README for web examples
    readme = """# Web Framework Examples

This directory contains examples of Python web frameworks.

## Flask (Microframework)

Flask is a lightweight WSGI web application framework.

### Features Demonstrated:
- Routing and views
- Templates with Jinja2
- SQLAlchemy integration
- User authentication with Flask-Login
- RESTful API endpoints
- Error handling

### To Run:
```bash
pip install -r requirements-web.txt
export FLASK_APP=web_framework_examples.py
export FLASK_ENV=development
flask run
```

### Endpoints:
- `GET /` - Home page with template
- `GET /api/posts` - JSON API for posts
- `POST /api/posts` - Create new post (authenticated)
- `GET /login` - Login page
- `POST /login` - Login endpoint
- `GET /create` - Create post page
- `POST /create` - Create post endpoint

## FastAPI (Async Framework)

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.6+ based on standard Python type hints.

### Features Demonstrated:
- Async/await support
- Automatic API documentation (OpenAPI/Swagger)
- Pydantic models for data validation
- SQLAlchemy integration
- JWT-like authentication (simplified)
- CRUD operations with pagination

### To Run:
```bash
pip install -r requirements-web.txt
uvicorn web_framework_examples:fastapi_app --reload
```

### Endpoints:
- `GET /` - Root endpoint
- `GET /docs` - Interactive API documentation
- `GET /users/` - Get users with pagination
- `POST /users/` - Create user
- `GET /posts/` - Get posts with filtering
- `POST /posts/` - Create post (authenticated)
- `PUT /posts/{post_id}` - Update post (authenticated)
- `DELETE /posts/{post_id}` - Delete post (authenticated)

## Authentication

Both examples include simplified authentication:
- Flask: Session-based login
- FastAPI: Bearer token authentication (simplified)

## Database

Both examples use SQLite for simplicity:
- Flask: `flask_example.db`
- FastAPI: `fastapi_example.db`

## Docker Deployment

Use Docker Compose to run both applications with PostgreSQL:

```bash
docker-compose up -d
```

## Notes

1. These are demonstration examples and not production-ready.
2. Password handling is simplified - use proper hashing in production.
3. Error handling is basic - add more robust error handling for production.
4. Consider using environment variables for configuration.
5. Add HTTPS, CORS, rate limiting, etc. for production deployments.
"""
    
    with open("WEB_FRAMEWORK_README.md", "w") as f:
        f.write(readme)
    
    print("\nCreated WEB_FRAMEWORK_README.md")
    
    # Environment configuration
    env_config = """# Web Application Configuration

# Flask Configuration
FLASK_APP=web_framework_examples.py
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-here-change-in-production

# Database
DATABASE_URL=sqlite:///app.db
# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/app_db

# FastAPI Configuration
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000
UVICORN_RELOAD=1

# Authentication
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30

# CORS (for development)
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
"""
    
    with open(".env.web", "w") as f:
        f.write(env_config)
    
    print("Created .env.web configuration file")


if __name__ == "__main__":
    main()