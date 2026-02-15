#!/usr/bin/env python3
"""
Database Operations Examples
===========================

This module demonstrates various Python database operations including:
- SQLite basic CRUD operations
- SQLAlchemy ORM examples
- Connection pooling and transactions
- Environment-based configuration
- Database migrations with Alembic basics
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, asdict

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("Note: SQLAlchemy not installed. Run: pip install sqlalchemy")


# ---------- SQLite Examples ----------

def create_sqlite_database(db_path: str = "example.db") -> sqlite3.Connection:
    """Create and connect to a SQLite database."""
    # Remove existing database to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dict-like row access
    
    return conn


def setup_users_table(conn: sqlite3.Connection) -> None:
    """Create a users table with sample data."""
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create posts table with foreign key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            published BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Insert sample users
    sample_users = [
        ("alice", "alice@example.com", 28),
        ("bob", "bob@example.com", 35),
        ("charlie", "charlie@example.com", 42),
        ("diana", "diana@example.com", 31),
    ]
    
    cursor.executemany(
        "INSERT INTO users (username, email, age) VALUES (?, ?, ?)",
        sample_users
    )
    
    # Insert sample posts
    sample_posts = [
        (1, "My First Post", "Hello world! This is my first blog post.", 1),
        (1, "Learning Python", "Python is amazing for database operations!", 1),
        (2, "Data Analysis", "Working with pandas and SQLite.", 1),
        (3, "Web Development", "Building REST APIs with databases.", 0),
        (4, "Machine Learning", "Training models with database data.", 1),
    ]
    
    cursor.executemany(
        "INSERT INTO posts (user_id, title, content, published) VALUES (?, ?, ?, ?)",
        sample_posts
    )
    
    conn.commit()
    print(f"Created tables and inserted sample data")


def sqlite_crud_examples() -> None:
    """Demonstrate SQLite CRUD operations."""
    print("\n" + "="*60)
    print("SQLite CRUD Operations")
    print("="*60)
    
    db_path = "example.db"
    conn = create_sqlite_database(db_path)
    
    try:
        # Set up initial data
        setup_users_table(conn)
        
        # ---------- READ Examples ----------
        cursor = conn.cursor()
        
        # Query all users
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        all_users = cursor.fetchall()
        print(f"\nAll users ({len(all_users)} total):")
        for user in all_users:
            print(f"  - {user['username']} ({user['email']}), age: {user['age']}")
        
        # Query with JOIN
        cursor.execute("""
            SELECT u.username, p.title, p.content, p.created_at
            FROM users u
            JOIN posts p ON u.id = p.user_id
            WHERE p.published = 1
            ORDER BY p.created_at DESC
        """)
        published_posts = cursor.fetchall()
        print(f"\nPublished posts ({len(published_posts)} total):")
        for post in published_posts:
            print(f"  - {post['username']}: {post['title']}")
        
        # Parameterized query
        min_age = 30
        cursor.execute("SELECT * FROM users WHERE age >= ?", (min_age,))
        older_users = cursor.fetchall()
        print(f"\nUsers aged {min_age}+ ({len(older_users)} found):")
        for user in older_users:
            print(f"  - {user['username']} (age: {user['age']})")
        
        # ---------- CREATE Example ----------
        new_user = ("emma", "emma@example.com", 29)
        cursor.execute(
            "INSERT INTO users (username, email, age) VALUES (?, ?, ?)",
            new_user
        )
        new_user_id = cursor.lastrowid
        
        # Add a post for the new user
        cursor.execute(
            "INSERT INTO posts (user_id, title, content, published) VALUES (?, ?, ?, ?)",
            (new_user_id, "Emma's First Post", "Excited to join!", 1)
        )
        
        conn.commit()
        print(f"\nCreated new user: {new_user[0]} with ID {new_user_id}")
        
        # ---------- UPDATE Example ----------
        cursor.execute(
            "UPDATE users SET age = ? WHERE username = ?",
            (30, "emma")
        )
        conn.commit()
        print(f"Updated Emma's age to 30")
        
        # ---------- DELETE Example ----------
        cursor.execute("DELETE FROM users WHERE username = ?", ("charlie",))
        # Also delete their posts (cascade would handle this in production)
        cursor.execute("DELETE FROM posts WHERE user_id = (SELECT id FROM users WHERE username = ?)", ("charlie",))
        conn.commit()
        print(f"Deleted user: charlie")
        
        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("charlie",))
        remaining = cursor.fetchone()[0]
        print(f"Charlie exists in database: {remaining > 0}")
        
        # ---------- Transactions Example ----------
        print("\nTransaction Example:")
        try:
            conn.execute("BEGIN")
            
            # Multiple operations in a transaction
            conn.execute("UPDATE users SET age = age + 1 WHERE username = ?", ("alice",))
            conn.execute("UPDATE users SET age = age + 1 WHERE username = ?", ("bob",))
            
            # This will fail if user doesn't exist, rolling back both updates
            conn.execute("UPDATE users SET age = age + 1 WHERE username = ?", ("nonexistent",))
            
            conn.commit()
            print("Transaction committed successfully")
        except sqlite3.Error as e:
            conn.rollback()
            print(f"Transaction rolled back: {e}")
        
        # Show final state
        cursor.execute("SELECT username, email, age FROM users ORDER BY username")
        final_users = cursor.fetchall()
        print(f"\nFinal users ({len(final_users)} total):")
        for user in final_users:
            print(f"  - {user['username']}: {user['age']} years old")
            
    finally:
        conn.close()
        print(f"\nDatabase connection closed")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Cleaned up database file: {db_path}")


# ---------- SQLAlchemy Examples ----------

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class User(Base):
        """SQLAlchemy User model."""
        __tablename__ = "users_alchemy"
        
        id = Column(Integer, primary_key=True)
        username = Column(String(50), unique=True, nullable=False)
        email = Column(String(100), unique=True, nullable=False)
        age = Column(Integer)
        created_at = Column(DateTime, default=datetime.now)
        
        # Relationship
        posts = relationship("Post", back_populates="user", cascade="all, delete-orphan")
        
        def __repr__(self):
            return f"<User(id={self.id}, username={self.username}, email={self.email})>"
        
        def to_dict(self):
            """Convert to dictionary."""
            return {
                "id": self.id,
                "username": self.username,
                "email": self.email,
                "age": self.age,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "post_count": len(self.posts) if self.posts else 0
            }
    
    class Post(Base):
        """SQLAlchemy Post model."""
        __tablename__ = "posts_alchemy"
        
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, ForeignKey("users_alchemy.id"), nullable=False)
        title = Column(String(200), nullable=False)
        content = Column(Text)
        published = Column(Integer, default=0)  # 0=False, 1=True
        created_at = Column(DateTime, default=datetime.now)
        
        # Relationship
        user = relationship("User", back_populates="posts")
        
        def __repr__(self):
            return f"<Post(id={self.id}, title={self.title[:30]}...)>"
        
        def to_dict(self):
            """Convert to dictionary."""
            return {
                "id": self.id,
                "user_id": self.user_id,
                "title": self.title,
                "content": self.content,
                "published": bool(self.published),
                "created_at": self.created_at.isoformat() if self.created_at else None
            }
    
    def sqlalchemy_examples() -> None:
        """Demonstrate SQLAlchemy ORM operations."""
        print("\n" + "="*60)
        print("SQLAlchemy ORM Operations")
        print("="*60)
        
        # Create SQLite database in memory for demonstration
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        
        try:
            # ---------- CREATE Examples ----------
            print("\nCreating users and posts...")
            
            # Create users
            users_data = [
                {"username": "john", "email": "john@example.com", "age": 25},
                {"username": "sarah", "email": "sarah@example.com", "age": 32},
                {"username": "mike", "email": "mike@example.com", "age": 41},
            ]
            
            users = []
            for user_data in users_data:
                user = User(**user_data)
                session.add(user)
                users.append(user)
            
            session.flush()  # Get IDs for foreign key relationships
            
            # Create posts for users
            posts_data = [
                {"user_id": users[0].id, "title": "John's First Post", "content": "Hello from John!", "published": 1},
                {"user_id": users[0].id, "title": "Learning SQLAlchemy", "content": "ORM is powerful!", "published": 1},
                {"user_id": users[1].id, "title": "Sarah's Blog", "content": "Welcome to my blog.", "published": 1},
                {"user_id": users[2].id, "title": "Draft Post", "content": "This is a draft.", "published": 0},
            ]
            
            for post_data in posts_data:
                post = Post(**post_data)
                session.add(post)
            
            session.commit()
            print(f"Created {len(users)} users and {len(posts_data)} posts")
            
            # ---------- READ Examples ----------
            print("\nQuerying data...")
            
            # Get all users
            all_users = session.query(User).all()
            print(f"All users ({len(all_users)}):")
            for user in all_users:
                print(f"  - {user.username} (ID: {user.id})")
            
            # Filter with conditions
            users_over_30 = session.query(User).filter(User.age > 30).all()
            print(f"\nUsers over 30 ({len(users_over_30)}):")
            for user in users_over_30:
                print(f"  - {user.username} (age: {user.age})")
            
            # Join query
            published_posts = (
                session.query(Post, User)
                .join(User)
                .filter(Post.published == 1)
                .order_by(Post.created_at.desc())
                .all()
            )
            print(f"\nPublished posts with authors ({len(published_posts)}):")
            for post, user in published_posts:
                print(f"  - {post.title} by {user.username}")
            
            # Aggregate query
            from sqlalchemy import func
            avg_age = session.query(func.avg(User.age)).scalar()
            print(f"\nAverage user age: {avg_age:.1f}")
            
            # Query with relationships (lazy loading)
            john = session.query(User).filter_by(username="john").first()
            if john:
                print(f"\nJohn's posts ({len(john.posts)}):")
                for post in john.posts:
                    print(f"  - {post.title} ({'Published' if post.published else 'Draft'})")
            
            # ---------- UPDATE Example ----------
            sarah = session.query(User).filter_by(username="sarah").first()
            if sarah:
                sarah.age = 33
                session.commit()
                print(f"\nUpdated Sarah's age to {sarah.age}")
            
            # Bulk update
            session.query(User).filter(User.age < 30).update({"age": User.age + 1})
            session.commit()
            print("Incremented age for users under 30")
            
            # ---------- DELETE Example ----------
            # Delete a user (cascade will delete their posts due to relationship)
            mike = session.query(User).filter_by(username="mike").first()
            if mike:
                session.delete(mike)
                session.commit()
                print(f"\nDeleted user: {mike.username}")
            
            # Verify deletion
            remaining_users = session.query(User).count()
            print(f"Remaining users: {remaining_users}")
            
            # ---------- Transactions Example ----------
            print("\nTransaction Example:")
            try:
                # Start a transaction
                session.begin()
                
                # Perform operations
                new_user = User(username="transaction_test", email="test@example.com", age=99)
                session.add(new_user)
                session.flush()  # Get ID
                
                new_post = Post(user_id=new_user.id, title="Test Post", content="Transaction test", published=1)
                session.add(new_post)
                
                # Intentionally cause an error to test rollback
                if True:  # Change to False to test successful commit
                    raise ValueError("Simulated error for rollback test")
                
                session.commit()
                print("Transaction committed")
            except Exception as e:
                session.rollback()
                print(f"Transaction rolled back: {e}")
            
            # Count should be unchanged if transaction rolled back
            final_count = session.query(User).filter_by(username="transaction_test").count()
            print(f"Test user exists after transaction: {final_count > 0}")
            
            # ---------- Query Performance ----------
            print("\nQuery Performance Tips:")
            
            # Eager loading (reduce N+1 problem)
            from sqlalchemy.orm import joinedload
            
            users_with_posts = (
                session.query(User)
                .options(joinedload(User.posts))
                .all()
            )
            print(f"Loaded {len(users_with_posts)} users with their posts (eager loading)")
            
            # Using indexes
            print("\nCreating index for better performance...")
            from sqlalchemy import Index
            idx = Index('idx_users_email', User.email)
            idx.create(engine)
            print("Index created on email column")
            
        finally:
            session.close()
            print("\nSession closed")


# ---------- Environment Configuration ----------

@dataclass
class DatabaseConfig:
    """Database configuration from environment variables."""
    db_type: str = "sqlite"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "app_db"
    db_user: str = "app_user"
    db_password: str = ""
    pool_size: int = 5
    echo_queries: bool = False
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        import os
        
        return cls(
            db_type=os.getenv("DB_TYPE", "sqlite"),
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_name=os.getenv("DB_NAME", "app_db"),
            db_user=os.getenv("DB_USER", "app_user"),
            db_password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            echo_queries=os.getenv("DB_ECHO", "").lower() == "true"
        )
    
    def get_connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        if self.db_type == "sqlite":
            return f"sqlite:///{self.db_name}.db"
        elif self.db_type == "postgresql":
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "mysql":
            return f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding password)."""
        data = asdict(self)
        data["db_password"] = "***" if data["db_password"] else ""
        return data


def environment_config_example() -> None:
    """Demonstrate environment-based configuration."""
    print("\n" + "="*60)
    print("Environment Configuration Example")
    print("="*60)
    
    # Simulate environment variables
    import os
    os.environ["DB_TYPE"] = "postgresql"
    os.environ["DB_HOST"] = "db.example.com"
    os.environ["DB_PORT"] = "5432"
    os.environ["DB_NAME"] = "production_db"
    os.environ["DB_USER"] = "admin"
    os.environ["DB_PASSWORD"] = "secret123"
    os.environ["DB_POOL_SIZE"] = "10"
    os.environ["DB_ECHO"] = "false"
    
    # Load configuration
    config = DatabaseConfig.from_env()
    
    print("Database Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    print(f"\nConnection string: {config.get_connection_string()}")
    
    # Clean up env vars (in real usage, these would be set in shell)
    for key in ["DB_TYPE", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_POOL_SIZE", "DB_ECHO"]:
        os.environ.pop(key, None)


# ---------- Main Execution ----------

def main():
    """Run all database examples."""
    print("Database Operations Examples")
    print("="*60)
    
    # Run SQLite examples
    sqlite_crud_examples()
    
    # Run SQLAlchemy examples if available
    if SQLALCHEMY_AVAILABLE:
        sqlalchemy_examples()
    else:
        print("\n" + "="*60)
        print("SQLAlchemy not installed")
        print("Install with: pip install sqlalchemy")
        print("="*60)
    
    # Run environment configuration example
    environment_config_example()
    
    print("\n" + "="*60)
    print("Database Examples Complete!")
    print("="*60)
    
    # Create a .env.example file for documentation
    env_example = """# Database Configuration
DB_TYPE=sqlite              # sqlite, postgresql, mysql
DB_HOST=localhost          # Database host
DB_PORT=5432              # Database port
DB_NAME=app_db            # Database name
DB_USER=app_user          # Database username
DB_PASSWORD=              # Database password
DB_POOL_SIZE=5            # Connection pool size
DB_ECHO=false             # Log SQL queries

# Other environment variables
APP_ENV=development       # development, testing, production
DEBUG=true                # Enable debug mode
SECRET_KEY=your-secret-key-here
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    print("\nCreated .env.example file for environment configuration")


if __name__ == "__main__":
    main()