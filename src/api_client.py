import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

def test_connection():
    response = requests.get(f"{BASE_URL}/status", headers=HEADERS)
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    test_connection()