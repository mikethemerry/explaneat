import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager


Base = declarative_base()


class Database:
    """Database connection manager similar to Flask-SQLAlchemy"""
    
    def __init__(self, database_url=None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'postgresql://localhost/explaneat_dev')
        self.engine = None
        self.session_factory = None
        self.Session = None
        
    def init_db(self, database_url=None):
        """Initialize database connection"""
        if database_url:
            self.database_url = database_url
            
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=NullPool,  # Good for scientific computing where connections might be held for long periods
            echo=os.getenv('SQLALCHEMY_ECHO', 'false').lower() == 'true'
        )
        
        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
        
    def create_all(self):
        """Create all tables"""
        if not self.engine:
            self.init_db()
        Base.metadata.create_all(self.engine)
        
    def drop_all(self):
        """Drop all tables"""
        if not self.engine:
            self.init_db()
        Base.metadata.drop_all(self.engine)
        
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations"""
        if not self.Session:
            self.init_db()
            
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            
    def get_session(self):
        """Get a new session"""
        if not self.Session:
            self.init_db()
        return self.Session()
        
    def close(self):
        """Close all sessions and dispose of engine"""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()


# Global database instance
db = Database()