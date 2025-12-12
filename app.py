import os
from PIL import Image
from dotenv import load_dotenv
from google import genai
from flask import Flask, request, render_template

# Load environment variables
load_dotenv()

# Configure Gemini
# No top-level configuration needed for new SDK Client pattern

app = Flask(__name__)

# Helper function to get the Gemini client
# Lazy loading prevents startup crashes if API key is missing
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)

@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = None
    error_message = None

    if request.method == 'POST':
        try:
            uploaded_file = request.files.get('file')
            if uploaded_file and uploaded_file.filename != '':
                image = Image.open(uploaded_file)

                # AI prompt
                prompt = (
                    "Analyze this uploaded medicine image. Provide accurate, clear, and useful medical details: "
                    "medicine name, uses, composition, dosage guidance, side effects, and other user-relevant info. "
                    "Also include a small note: 'Consult a doctor before use.'"
                )

                # Initialize client and generate content
                # Uses the model from environment or defaults to gemini-2.0-flash
                # Confirmed available model from diagnostic: gemini-2.0-flash
                model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                client = get_gemini_client()
                
                # New SDK syntax
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image]
                )
                
                response_text = response.text if response.text else "No description generated."
            else:
                error_message = "Please upload a valid image file."

        except Exception as e:
            # Catch errors (like Quota exceeded, Invalid API Key, Model not found) 
            # and display them on the page instead of crashing the server (500 error)
            error_message = f"Error: {str(e)}"

    return render_template('index.html', response_text=response_text, error=error_message)

if __name__ == '__main__':
    app.run(debug=True, port=3000)