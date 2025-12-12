import os
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, render_template

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

app = Flask(__name__)

# Initialize model
model = genai.GenerativeModel(MODEL_NAME)

@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            image = Image.open(uploaded_file)

            # AI prompt
            prompt = (
                "Analyze this uploaded medicine image. Provide accurate, clear, and useful medical details: "
                "medicine name, uses, composition, dosage guidance, side effects, and other user-relevant info. "
                "Also include a small note: 'Consult a doctor before use.'"
            )

            response = model.generate_content([prompt, image])
            response_text = response.text or "No description generated."

    return render_template('index.html', response_text=response_text)

if __name__ == '__main__':
    app.run(debug=True, port=3000)