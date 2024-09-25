import os
from flask_migrate import upgrade, init, migrate
from backend import create_app, db
from models.neat_model import NEATModel

app = create_app()


def init_db():
    # Check if migrations directory exists
    if not os.path.exists("migrations"):
        # Initialize migrations
        init()

    # Create a migration
    migrate()

    # Apply the migration
    upgrade()

    # Add some initial data if needed
    with app.app_context():
        # Check if the database is empty
        if NEATModel.query.count() == 0:
            # Add a sample NEAT model
            sample_model = NEATModel(
                model_name="Sample NEAT Model",
                dataset="Sample Dataset",
                version="1.0",
                raw_data=b"Sample raw data",
                parsed_model={"sample": "data"},
            )
            db.session.add(sample_model)
            db.session.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
