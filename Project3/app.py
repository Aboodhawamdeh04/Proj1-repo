import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select, insert
from db import engine
from models import destinations_table

# ---------------------------------------------------------
# 1. Load the AI Model (Cached for performance)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# ---------------------------------------------------------
# 2. Seed Database using Pure SQLAlchemy Core
# ---------------------------------------------------------
def seed_database():
    with engine.begin() as connection:
        # Check if table already has data using a Core select statement
        existing = connection.execute(select(destinations_table).limit(1)).first()
        if existing:
            return

        destinations = [
            {"name": "Munich", "country": "Germany", "description": "Famous for its automotive engineering history, cutting-edge technology museums, and beautiful Bavarian architecture.", "category": "City"},
            {"name": "San Francisco", "country": "USA", "description": "A massive hub for artificial intelligence, technology startups, and widespread electric vehicle adoption.", "category": "City"},
            {"name": "Rome", "country": "Italy", "description": "Ancient ruins and massive colosseums steeped in the rich history of ancient combat and gladiator battles.", "category": "Culture"},
            {"name": "Tuscany", "country": "Italy", "description": "Rolling hills renowned for their deep agricultural roots, traditional farming estates, and scenic countryside.", "category": "Nature"},
            {"name": "Tokyo", "country": "Japan", "description": "Neon-lit streets blending ancient shrines with advanced robotics, electronics, and digital gaming culture.", "category": "City"},
            {"name": "Kyoto", "country": "Japan", "description": "Historic city featuring ancient samurai districts, serene bamboo forests, and traditional wooden houses.", "category": "Culture"},
            {"name": "Swiss Alps", "country": "Switzerland", "description": "Towering peaks offering intense physical outdoor fitness challenges, steep running trails, and winter sports.", "category": "Adventure"},
            {"name": "Yellowstone", "country": "USA", "description": "Vast natural landscapes, geothermal geysers, and protected wildlife roaming the open plains.", "category": "Nature"},
            {"name": "Bali", "country": "Indonesia", "description": "Tropical beaches, terraced rice paddies, and a vibrant local culture perfect for relaxation.", "category": "Beach"},
            {"name": "Maldives", "country": "Maldives", "description": "Crystal clear waters and overwater bungalows offering the ultimate relaxing beach paradise.", "category": "Beach"},
            {"name": "Paris", "country": "France", "description": "Famous for classical art, massive museums, and iconic historical landmarks along the river.", "category": "Culture"},
            {"name": "Dubai", "country": "UAE", "description": "Modern city nightlife, towering skyscrapers, and luxury shopping in a rapidly developing metropolis.", "category": "City"},
            {"name": "Phuket", "country": "Thailand", "description": "Lively beaches, vibrant nightlife, and beautiful island-hopping boat tours.", "category": "Beach"},
            {"name": "Banff", "country": "Canada", "description": "Majestic rocky mountains and turquoise glacial lakes perfect for challenging mountain hiking.", "category": "Adventure"},
            {"name": "Petra", "country": "Jordan", "description": "A stunning ancient city carved directly into vibrant red sandstone cliffs, rich in architectural and trade history.", "category": "Culture"}
        ]
        
        for dest in destinations:
            embedding_vector = model.encode(dest["description"])
            embedding_bytes = embedding_vector.tobytes() 
            
            connection.execute(
                insert(destinations_table).values(
                    name=dest["name"],
                    country=dest["country"],
                    description=dest["description"],
                    category=dest["category"],
                    embedding=embedding_bytes
                )
            )

# Initialize data on startup
seed_database()

# ---------------------------------------------------------
# 3. Database Query Helper (Pure Core)
# ---------------------------------------------------------
def get_destinations_from_db():
    with engine.connect() as connection:
        result = connection.execute(select(destinations_table)).fetchall()
        return result

# ---------------------------------------------------------
# 4. Semantic Search Engine
# ---------------------------------------------------------
def find_similar(user_input, destinations):
    input_embedding = model.encode(user_input).reshape(1, -1)
    
    scores = []
    for d in destinations:
        stored_embedding = np.frombuffer(d.embedding, dtype=np.float32).reshape(1, -1)
        score = cosine_similarity(input_embedding, stored_embedding)[0][0]
        scores.append((d, score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# ---------------------------------------------------------
# 5. Build the Streamlit UI
# ---------------------------------------------------------
st.title("🌍 AI Travel Recommendation App (SQLAlchemy Core)")
st.write("Describe your perfect vacation, and AI will find the best matches using SQLAlchemy Core!")

user_query = st.text_input("Enter your preferences (e.g., 'I want a relaxing beach vacation'):")

if st.button("Search"):
    if user_query:
        destinations = get_destinations_from_db()
        
        if not destinations:
            st.error("The database is empty! Please run your seeding script first.")
        else:
            results = find_similar(user_query, destinations)
            highest_score = results[0][1]
            
            if highest_score < 0.25:
                st.warning("No exact match found. Showing popular destinations instead.")
                
                fallback_results = destinations[:3] 
                for d in fallback_results:
                    st.subheader(f"{d.name}, {d.country}")
                    st.write(f"**Category:** {d.category}")
                    st.write(d.description)
                    st.divider()
            else:
                st.success("Here are your top recommendations:")
                
                for d, score in results[:3]:
                    st.subheader(f"{d.name}, {d.country}")
                    st.write(f"**Category:** {d.category}")
                    st.write(d.description)
                    st.write(f"*Match Score: {score:.2f}*")
                    st.divider()