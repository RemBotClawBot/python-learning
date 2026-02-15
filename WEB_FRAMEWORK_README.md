# Web Framework Examples

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
