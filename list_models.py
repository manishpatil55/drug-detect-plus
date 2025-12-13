import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        exit(1)
        
    client = genai.Client(api_key=api_key)
    print("Listing available models...")
    
    # Try to list models
    # Note: methods might verify depending on exact SDK version, trying standard list
    pager = client.models.list()
    
    print("\n--- Available Models ---")
    for model in pager:
        # print(model) 
        # Check if it supports generation
        if 'generateContent' in model.supported_generation_methods:
            print(f"Name: {model.name}")
            print(f"Distplay Name: {model.display_name}")
            print("-" * 20)

except Exception as e:
    print(f"Error listing models: {e}")
