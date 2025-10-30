<a href="https://drug-detect-plus.vercel.app/">
  <img 
    src="public/drug-detect-ai.jpeg"
    alt="Drug Detect++ Preview"
    width="100%"
    style="border-radius: 12px; box-shadow: 0 0 25px rgba(0, 200, 200, 0.2);"
  />
</a>


# 💊 Drug Detect AI — Smart Medicine Image Analyzer

> AI-powered tool that identifies medicines just by their image.  
> Upload a medicine photo, and the AI tells you what it is, how it's used, dosage info, and more — powered by **Google Gemini AI**.

---

## 🚀 Features

- 🧠 **AI-Powered Analysis** — Uses Gemini AI to detect and describe medicines from images  
- 🖼️ **Image Upload Support** — Upload or drag-and-drop medicine photos  
- ⚡ **Instant Results** — Get AI responses with uses, dosage, and details  
- 💾 **Smart Preview Persistence** — Keeps your uploaded image visible even after reload  
- 🪄 **Modern UI** — Glassmorphic, animated background with glowing waves & orbs  
- 📱 **Responsive Design** — Works smoothly on desktop and mobile

---

## 🧩 Tech Stack

| Category | Technologies Used |
|-----------|-------------------|
| Frontend | HTML5, CSS3 (Custom UI, Glassmorphism, Animations), JavaScript |
| Backend | Python (Flask / FastAPI based) |
| AI Model | Google Gemini 2.0 Flash |
| Hosting Ready | Works locally or can be deployed on Vercel / Render / Replit |
| Extras | LocalStorage preview save, smooth scroll animations, copy-to-clipboard |

---

## ⚙️ Setup Guide

### 1️⃣ Clone this repo
```bash
git clone https://github.com/manishpatil55/drug-detect-plus.git
cd drug-detect-plus
```

2️⃣ Create a virtual environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate   # (Mac/Linux)
```
```bash
.venv\Scripts\activate      # (Windows)
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Create a .env file in the project root

Add your Gemini API key and model here 👇
You can get your Gemini API key from Google AI Studio.
```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

5️⃣ Run the app
```bash
python app.py
```

Then open your browser and go to:
```bash
http://localhost:5000
```

---

## 🧠 How It Works
   - You upload a photo of a medicine 📸
	- The backend sends the image to Gemini AI via your API key 🤖
	- Gemini analyzes and identifies the medicine 💊
	- The app shows:
   - Medicine Name
   - Uses and Composition
   - Dosage Info
   - Side Effects (if applicable)
   - External link suggestions for ordering

---

## 🪩 Demo Preview

![App Preview](public/drug-detect-ai.jpeg)

✨ A clean, glowing glassmorphic UI with smooth animations and orbs in the background.
Type-safe, scroll-aware, and built with minimal JS for performance.

---

🧑‍💻 Developer Notes
   - You can customize the AI prompt in app.py to change how detailed responses should be.
	- The entire app is front-end + Flask — no database required.
	- The localStorage stores preview image data for smooth reload experience.
	- Make sure .env is properly loaded before running the app.

---


📜 License

This project is open-source under the MIT License — free to use, remix, and improve!

---