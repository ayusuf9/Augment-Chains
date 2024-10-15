from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",  # React app runs on this port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
client = AsyncIOMotorClient('mongodb://localhost:27017')
db = client.mydatabase
collection = db.mycollection

@app.get("/data")
async def get_data():
    # Fetch data from a Python package (generate a random number)
    data = {"number": random.randint(1, 100)}
    # Save data to MongoDB
    await collection.insert_one(data)
    return data

@app.get("/data/{item_id}")
async def read_data(item_id: str):
    item = await collection.find_one({"_id": item_id})
    return item
