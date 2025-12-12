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
@limiter.limit("10 per minute") # Strict limit for the main analysis endpoint
def index():
    response_text = None
    error_message = None

    if request.method == 'POST':
        try:
            uploaded_file = request.files.get('file')
            
            # --- SECURITY CHECKS ---
            if not uploaded_file or uploaded_file.filename == '':
                error_message = "Please upload a valid image file."
            
            else:
                # Check 1: File Size (approximate check via seek/tell or content-length)
                # Note: valid only if client sends content-length, or we read the blob.
                # For robust check, we can check blob size after basic read.
                uploaded_file.seek(0, os.SEEK_END)
                file_size_mb = uploaded_file.tell() / (1024 * 1024)
                uploaded_file.seek(0) # Reset cursor

                if file_size_mb > MAX_FILE_SIZE_MB:
                     error_message = f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
                
                # Check 2: Mime Type
                elif uploaded_file.mimetype not in ALLOWED_MIMETYPES:
                    error_message = "Invalid file type. Only JPG, PNG, and WebP are allowed."

                else:
                    # File is safe enough to process
                    image = Image.open(uploaded_file)

                    # AI prompt
                    prompt = (
                        "Analyze this uploaded medicine image. Provide accurate, clear, and useful medical details: "
                        "medicine name, uses, composition, dosage guidance, side effects, and other user-relevant info. "
                        "Also include a small note: 'Consult a doctor before use.'"
                    )

                    # Initialize client and generate content
                    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
                    client = get_gemini_client()
                    
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image]
                    )
                    
                    response_text = response.text if response.text else "No description generated."

        except Exception as e:
            # Handle RateLimitExceeded specifically if needed, otherwise general catch
            if "429" in str(e):
                error_message = "Too many requests. Please try again in a minute."
            else:
                error_message = f"Error: {str(e)}"

    return render_template('index.html', response_text=response_text, error=error_message)

# Handle Rate Limit Errors specifically to show nice message
@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template('index.html', error="You are sending requests too fast. Please wait a moment."), 429

if __name__ == '__main__':
    app.run(debug=True, port=3000)