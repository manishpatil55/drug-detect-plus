import os
import time
from PIL import Image
from dotenv import load_dotenv
from google import genai
from flask import Flask, request, render_template, jsonify

# Security Imports
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- SECURITY CONFIGURATION ---
# Rate Limiting DISABLED per user request
# limiter = Limiter(
#     get_remote_address,
#     app=app,
#     default_limits=["1000 per day", "10 per minute"],
#     storage_uri="memory://"
# )

# 2. File Validation Config
MAX_FILE_SIZE_MB = 10
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}

# --- HELPER: Multi-Key Rotation Logic ---
def generate_content_with_rotation(model_name, contents):
    """
    Rotates through comma-separated GEMINI_API_KEYs on 429 errors.
    Returns: response object
    Raises: Exception if all keys fail.
    """
    raw_keys = os.getenv("GEMINI_API_KEY", "")
    if not raw_keys:
        raise ValueError("GEMINI_API_KEY not set")
    
    api_keys = [k.strip() for k in raw_keys.split(',') if k.strip()]
    if not api_keys:
         raise ValueError("No valid keys found in GEMINI_API_KEY")

    last_error = None
    
    for i, key in enumerate(api_keys):
        try:
            # Create client for this specific key
            client = genai.Client(api_key=key)
            
            # Attempt generation
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            return response # Success!
            
        except Exception as e:
            last_error = e
            # Only rotate if it's a Quota/Rate limit error (429)
            if "429" in str(e):
                print(f"⚠️ Key {i+1}/{len(api_keys)} exhausted (429). Rotating to next key...")
                continue # Try next key
            else:
                # If it's a model not found or other fatal error, typical retry won't help
                # UNLESS it's a server error (500), but generic '429' check is safest for quota.
                # Currently we only catch 429 for rotation.
                raise e

    # If we exit loop, all keys failed
    raise last_error

@app.route('/', methods=['GET', 'POST'])
# @limiter.limit("10 per minute") # Strict limit
def index():
    response_text = None
    error_message = None

    if request.method == 'POST':
        try:
            # 1. Handle Multiple Files
            uploaded_files = request.files.getlist('file')
            
            valid_images = []
            
            # --- SECURITY & VALIDATION LOOP ---
            if not uploaded_files or uploaded_files[0].filename == '':
                 error_message = "Please upload at least one valid image."
            else:
                for file in uploaded_files:
                    # Check Size
                    file.seek(0, os.SEEK_END)
                    size_mb = file.tell() / (1024 * 1024)
                    file.seek(0)
                    
                    if size_mb > MAX_FILE_SIZE_MB:
                        error_message = f"File {file.filename} is too large (> {MAX_FILE_SIZE_MB}MB)."
                        break
                    
                    # Check Type
                    if file.mimetype not in ALLOWED_MIMETYPES:
                        error_message = f"File {file.filename} has invalid type. Only images allowed."
                        break
                    
                    valid_images.append(Image.open(file))
            
            # If no errors and we have images, proceed to AI
            if not error_message and valid_images:
                
                # Dynamic Prompt Construction
                prompt_parts = [
                    "Analyze the uploaded medicine image(s).",
                    "For **EACH** distinct medicine detected, output the following section:",
                    "",
                    "### <Medicine Name>",
                    "",
                    "**Analysis Table**",
                    "Create a Markdown Table with columns 'Feature' and 'Details'.",
                    "Rows: **Usage**, **Dosage** (Adult/Child), **Side Effects** (Common/Rare), **Mechanism**.",
                    "",
                    "**Where to Buy**",
                    "Output these 4 links on a **SINGLE LINE** separated by standard spaces.",
                    "**IMPORTANT**: In the URLs, replace any spaces in the medicine name with `+` (e.g., `Calpol+500`). Do NOT leave actual spaces in the URL.",
                    "Format: `[Buy on Vendor](URL)` (No space between brackets and parentheses).",
                    "- [Buy on 1mg](https://www.1mg.com/search/all?name=<Name_with_+>)",
                    "- [Buy on Apollo](https://www.apollopharmacy.in/search-medicines/<Name_with_+>)",
                    "- [Buy on Netmeds](https://www.netmeds.com/catalogsearch/result?q=<Name_with_+>)",
                    "- [Buy on Amazon](https://www.amazon.in/s?k=<Name_with_+>&tag=drugdetectai-21)",
                    "",
                    "---",
                    "",
                    "(After listing all medicines):",
                    "### ⚠️ Drug Interactions",
                    "Checking if these medicines can be taken together...",
                    "",
                    "### ℹ️ Disclaimer",
                    "'Consult a doctor before use.'"
                ]
                
                full_prompt = "\n".join(prompt_parts)

                # Initialize client & Generate with Rotation
                model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
                
                response = generate_content_with_rotation(
                    model_name=model_name, 
                    contents=[full_prompt] + valid_images
                )
                
                response_text = response.text if response.text else "No description generated."


        except Exception as e:
            if "429" in str(e):
                error_message = (
                    "🚨 **Daily Quota Exceeded**.<br>"
                    "Your free API key has hit its limit (20/day) for this model.<br>"
                    "**Fix:** Create a NEW Google API Key (in a new project) to get 1500 requests/day.<br>"
                    f"<small>Details: {str(e)}</small>"
                )
            else:
                error_message = f"Error: {str(e)}"

    # Check for AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'response_text': response_text, 'error': error_message})

    return render_template('index.html', response_text=response_text, error=error_message)

@app.route('/translate', methods=['POST'])
# @limiter.limit("20 per minute")
def translate_text():
    try:
        data = request.json
        text = data.get('text')
        target_lang = data.get('language')
        
        if not text or not target_lang:
            return jsonify({'error': 'Invalid request'}), 400

        # Translation Prompt
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        prompt = f"Translate the following medical analysis to {target_lang}. Maintain all Markdown formatting (bold, headers, links) exactly as is. Output only the translated text.\n\n{text}"
        
        # Call with Rotation
        response = generate_content_with_rotation(
            model_name=model_name,
            contents=[prompt]
        )
        translated_text = response.text

        return jsonify({'translated_text': translated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handle Rate Limit Errors specifically to show nice message
@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template('index.html', error="You are sending requests too fast. Please wait a moment."), 429

if __name__ == '__main__':
    app.run(debug=True, port=3000)