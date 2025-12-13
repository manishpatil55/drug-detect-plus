import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-2.0-flash-exp",
    "gemini-exp-1206",
    "gemini-1.5-flash",
]

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Fallback to manual load if dotenv fails in this env (redundancy)
    try:
        with open('.env') as f:
            for line in f:
                if line.strip() and "GEMINI_API_KEY" in line:
                    api_key = line.split('=')[1].strip().strip('"').strip("'")
                    break
    except:
        pass

if not api_key:
    print("NO KEY FOUND")
    exit(1)

client = genai.Client(api_key=api_key)

print(f"Testing {len(CANDIDATES)} models for availability/quota...")

for model in CANDIDATES:
    print(f"\nTesting: {model} ...", end=" ", flush=True)
    try:
        response = client.models.generate_content(
            model=model,
            contents="Hello, barely use any tokens."
        )
        print("✅ SUCCESS!")
        print(f"Response: {response.text}")
        print(f"!!! MODEL AVAILABLE: {model} !!!")
        # Continue testing to find others
        # break
    except Exception as e:
        print("❌ FAILED")
        err_str = str(e)
        if "404" in err_str:
            print("   -> Not Found")
        elif "429" in err_str:
            print("   -> Rate/Quota Limit")
            # If it's a limit, print details to see if it's 0 or just exhausted
            if "limit: 0" in err_str:
                 print("   -> Limit is ZERO (Not available)")
            else:
                 print("   -> Quota exhausted (but exists)")
        else:
            print(f"   -> Error: {err_str[:100]}...")
