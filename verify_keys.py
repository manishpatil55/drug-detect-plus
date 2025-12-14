import os
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

def verify_keys():
    print("🔐 Loading keys from GEMINI_API_KEY...")
    
    api_keys_str = os.getenv("GEMINI_API_KEY", "")
    if not api_keys_str:
        print("❌ Error: GEMINI_API_KEY is empty.")
        return

    keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
    model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    
    print(f"📋 Found {len(keys)} keys.")
    print(f"🎯 Target Model: {model_id}")
    print("-" * 40)

    for i, key in enumerate(keys):
        masked_key = f"{key[:5]}...{key[-3:]}"
        print(f"Testing Key {i+1}: {masked_key}", end=" ")
        
        try:
            # Metadata check only (Should NOT consume Generation Quota)
            client = genai.Client(api_key=key)
            model_info = client.models.get(model=f"models/{model_id}")
            
            print(f"✅ SUCCESS")
            print(f"   Model Name: {model_info.display_name}")
            print(f"   Input Limit: {model_info.input_token_limit} tokens")
            
        except Exception as e:
            print(f"❌ FAILED")
            # print(f"   Error: {str(e)}")
            if "404" in str(e):
                print("   Error: Model Not Found (Check if this key supports this model)")
            elif "403" in str(e):
                print("   Error: Permission Denied (API Key might be invalid)")
            else:
                print(f"   Error: {str(e)}")

        print("-" * 40)
        time.sleep(1) # Be gentle

if __name__ == "__main__":
    verify_keys()
