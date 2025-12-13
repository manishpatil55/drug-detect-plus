import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        exit(1)
        
    client = genai.Client(api_key=api_key)
    print("Successfully connected. Listing models...")
    
    # List models via standard method
    try:
        pager = client.models.list()
        print("\n--- AVAILABLE MODELS ---")
        for m in pager:
            # We filter for verify/generate capabilities if possible, or just print all
            if hasattr(m, 'name'):
                print(f"ID: {m.name}")
                if hasattr(m, 'display_name'):
                    print(f"Name: {m.display_name}")
                print("-" * 10)
    except Exception as e:
        print(f"Error calling client.models.list(): {e}")

except Exception as e:
    print(f"Critical Error: {e}")
