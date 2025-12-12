import os
from PIL import Image
from dotenv import load_dotenv
from google import genai
from flask import Flask, request, render_template

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
    default_limits=["200 per day", "50 per hour"],
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
            # Default to English for initial scan
            language = "English"
            
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
                
                # Dynamic Prompt Construction (Always English Initial)
                prompt_parts = [
                    "Analyze the uploaded medicine image(s). Provide response in **English**.",
                    "1. Identify the medicine(s) name, usage, dosage, and side effects.",
                    "2. **Critical**: If multiple different medicines are visible, check for **DRUG INTERACTIONS** between them. Can they be taken together?",
                    "3. **Affiliate/Buying**: For each identified medicine, generate a 'Buy Online' section.",
                    "   - Provide search links to: **1mg**, **Apollo Pharmacy**, **Netmeds**, **Amazon**.",
                    "   - Format: `[Buy on 1mg](https://www.1mg.com/search/all?name=<Medicine Name>)`",
                    "4. Disclaimer: 'Consult a doctor before use.'"
                ]
                
                full_prompt = "\n".join(prompt_parts)

                # Initialize client
                model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
                client = get_gemini_client()
                
                # Pass *List* of images + Prompt
                response = client.models.generate_content(
                    model=model_name,
                    contents=[full_prompt] + valid_images
                )
                
                response_text = response.text if response.text else "No description generated."

        except Exception as e:
            if "429" in str(e):
                error_message = "Too many requests. Please try again in a minute."
            else:
                error_message = f"Error: {str(e)}"

    return render_template('index.html', response_text=response_text, error=error_message)

@app.route('/translate', methods=['POST'])
@limiter.limit("20 per minute") # Higher limit for translations
def translate_text():
    try:
        data = request.get_json()
        text_to_translate = data.get('text')
        target_language = data.get('target_language')
        
        if not text_to_translate or not target_language:
             return {"error": "Missing text or target_language"}, 400

        # Translation Prompt
        prompt = f"Translate the following medical text to **{target_language}**. Preserve all Markdown formatting (bold, list, links) exactly. Do not add any new content, just translate.\n\nText:\n{text_to_translate}"
        
        client = get_gemini_client()
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        
        response = client.models.generate_content(
             model=model_name,
             contents=[prompt]
        )
        
        return {"translated_text": response.text}

    except Exception as e:
        return {"error": str(e)}, 500

# Handle Rate Limit Errors specifically to show nice message
@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template('index.html', error="You are sending requests too fast. Please wait a moment."), 429

if __name__ == '__main__':
    app.run(debug=True, port=3000)