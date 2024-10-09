from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .config import Config

db = SQLAlchemy()
migrate = Migrate()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": "http://localhost:3000",  # Replace with your frontend URL if different
                "supports_credentials": True,
            }
        },
    )
    db.init_app(app)
    migrate.init_app(app, db)

    from .routes.api import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    # Create database tables
    with app.app_context():
        db.create_all()

    return app
