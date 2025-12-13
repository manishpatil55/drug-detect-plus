import os
from PIL import Image
from dotenv import load_dotenv
from google import genai
from flask import Flask, request, render_template, jsonify

# Security Imports
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- SECURITY CONFIGURATION ---
# 1. Rate Limiting: Restrict IP to 10 requests per minute to prevent spam/quota drain
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1000 per day", "10 per minute"],
    storage_uri="memory://"
)

# 2. File Validation Config
MAX_FILE_SIZE_MB = 10
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}

# Helper function to get the Gemini client
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute") # Strict limit
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
                    "### 💊 Medicine Analysis",
                    "Identify the medicine(s) name, usage, dosage, and side effects.",
                    "### ⚠️ Drug Interactions",
                    "**Critical**: If multiple different medicines are visible, check if they can be taken together. Explain any risks.",
                    "### 🛒 WHERE TO BUY",
                    "For each identified medicine, generate a 'Buy Online' section.",
                    "   - Provide search links to: **1mg**, **Apollo Pharmacy**, **Netmeds**, **Amazon**.",
                    "   - Format: `[Buy on 1mg](https://www.1mg.com/search/all?name=<Medicine Name>)`",
                    "### ℹ️ Disclaimer",
                    "'Consult a doctor before use.'"
                ]
                
                full_prompt = "\n".join(prompt_parts)

                # Initialize client
                model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
                client = get_gemini_client()
                
                # Retry Logic (3 Attempts) to handle 429s
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[full_prompt] + valid_images
                        )
                        response_text = response.text if response.text else "No description generated."
                        break # Success
                    except Exception as try_err:
                        # If it's the last attempt or not a 429, re-raise to outer block
                        if attempt == 2 or "429" not in str(try_err):
                            raise try_err
                        time.sleep(2) # Wait 2 seconds before retry

        except Exception as e:
            if "429" in str(e):
                error_message = "Too many requests. Please try again in a minute."
            else:
                error_message = f"Error: {str(e)}"

    return render_template('index.html', response_text=response_text, error=error_message)

@app.route('/translate', methods=['POST'])
@limiter.limit("20 per minute")
def translate_text():
    try:
        data = request.json
        text = data.get('text')
        target_lang = data.get('language')
        
        if not text or not target_lang:
            return jsonify({'error': 'Invalid request'}), 400

        # Translation Prompt
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        client = get_gemini_client()
        
        prompt = f"Translate the following medical analysis to {target_lang}. Maintain all Markdown formatting (bold, headers, links) exactly as is. Output only the translated text.\n\n{text}"
        
        # Retry Logic (3 Attempts) for Translation
        translated_text = ""
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt]
                )
                translated_text = response.text
                break
            except Exception as try_err:
                 if attempt == 2 or "429" not in str(try_err):
                    raise try_err
                 time.sleep(2)

        return jsonify({'translated_text': translated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handle Rate Limit Errors specifically to show nice message
@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template('index.html', error="You are sending requests too fast. Please wait a moment."), 429

if __name__ == '__main__':
    app.run(debug=True, port=3000)