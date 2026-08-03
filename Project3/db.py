from sqlalchemy import create_engine

# Update with your actual PostgreSQL credentials
DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost/travel_rec_db"
engine = create_engine(DATABASE_URL)