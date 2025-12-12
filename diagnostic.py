import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# Test with 1.5-flash as it's the most reliable
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

print(f"DEBUG: Checking API Key for NEW SDK...")
if not api_key:
    print("ERROR: GEMINI_API_KEY is missing!")
else:
    print(f"DEBUG: API Key found (starts with {api_key[:4]}...)")

print(f"DEBUG: Configuring NEW SDK Client...")
try:
    client = genai.Client(api_key=api_key)
    print(f"DEBUG: Client initialized. Listing available models...")
    
    # List all available models
    for m in client.models.list():
        print(f"FOUND MODEL: {m.name}")

except Exception as e:
    print(f"CRITICAL ERROR: {str(e)}")
