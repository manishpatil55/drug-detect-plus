
<div align="center">
  <a href="https://drug-detect-plus.vercel.app/">
    <img src="public/drug-detect-ai.jpeg" alt="Drug Detect AI - Smart Medicine Analyzer" width="100%" style="border-radius: 16px; box-shadow: 0 8px 30px rgba(0, 150, 150, 0.3); border: 1px solid rgba(255,255,255,0.1);" />
  </a>

  <br />
  <br />

  <h1>💊 Drug Detect AI</h1>
  <p class="lead"><strong>Smart Medicine Analyzer with Hybrid Search & Market Intelligence</strong></p>

  <p>
    <a href="https://drug-detect-plus.vercel.app/"><strong>View Live Demo</strong></a> • 
    <a href="#-setup-guide"><strong>Deploy Your Own</strong></a> • 
    <a href="#-license"><strong>License</strong></a>
  </p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Status-Production-success?style=for-the-badge" />
    <img src="https://img.shields.io/badge/AI-Gemini%202.5%20Flash-blue?style=for-the-badge" />
    <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-orange?style=for-the-badge" />
  </p>
</div>

---

## 🚀 Overview

**Drug Detect AI** is an advanced medical intelligence tool that allows users to identify medicines via **Image Upload** or **Text Search**. Powered by **Google Gemini 2.5 Flash**, it provides instant, detailed breakdowns of medicine usage, dosage, side effects, and mechanism of action.

Unlike standard analyzers, this tool features **Market Intelligence**: it automatically generates direct "Where to Buy" links for top pharmacies (1mg, Apollo, Netmeds, Amazon) and checks for potential drug interactions.

---

## ✨ Key Features

### 🔍 Hybrid Search Engine
- **Image Analysis**: Upload a photo of a strip, bottle, or tablet. The AI reads text, shapes, and colors to identify the drug.
- **Smart Text Search**: Type a name (e.g., "Dolo 650") or a symptom (e.g., "headache medicine"). Handles typos and fuzzy matching automatically.

### 🧠 Advanced Medical Insights
- **Structured Data**: Returns a clean **Table View** of Uses, Dosage, Mechanism, and Side Effects.
- **Drug Interactions**: If multiple medicines are visible, it warns about potentially dangerous combinations.
- **Multi-Language Support**: Capable of analyzing generic names and local brands.

### 💸 Monetization & Affiliate Ready
- **Buy Buttons**: Automatically generates direct search links for **1mg, Apollo Pharmacy, Netmeds, and Amazon**.
- **Affiliate Integration**: Built-in support for Amazon Affiliate tags (configurable).

### 🎨 Premium UI/UX ("Glassmorphism")
- **Visuals**: Deep green ambient aesthetic with flowing particle backgrounds.
- **Responsive**: Fully optimized for Mobile (iOS/Android) and Desktop.
- **Interaction**: Micro-animations, pill-shaped glass buttons, and smooth loading states.

### 🛡️ Enterprise-Grade Reliability
- **API Key Rotation**: Backend automatically cycles through multiple API keys to handle rate limits (Quota Management).
- **Robust Error Handling**: Catches "Resource Exhausted" errors and retries seamlessly.

---

## 🧩 Tech Stack

| Component | Technology |
|:--- |:--- |
| **Frontend** | HTML5, CSS3 (Custom Glassmorphism), Vanilla JS |
| **Backend** | Python (Flask), Werkzeug |
| **AI Core** | Google Gemini 2.5 Flash Lite (via `google-genai` SDK) |
| **Monetization** | Dynamic Affiliate Link Generation |
| **Deployment** | Vercel (Python Runtime) |

---

## ⚙️ Setup Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/manishpatil55/drug-detect-plus.git
cd drug-detect-plus
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Environment
Create a `.env` file in the root directory:
```bash
# Comma-separated keys for auto-rotation/reliability
GEMINI_API_KEY=your_key_1,your_key_2,your_key_3

# AI Model (Optimized for Speed/Cost)
GEMINI_MODEL=gemini-2.5-flash-lite
```

### 4️⃣ Run Locally
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 📜 License

**Copyright © 2025 Manish Patil.**

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

✅ **You CAN**:
- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material.

❌ **You CANNOT**:
- **Commercial Use**: You may not use the material for commercial purposes (selling the app, paid services) without explicit permission from the author.
- **No Attribution**: You must give appropriate credit to the original author.

---

## 📞 Contact & Credits

**Developed by [Manish Patil](https://github.com/manishpatil55)**.

If you wish to use this software for commercial purposes or have feature requests, please contact the developer via GitHub Issues.

---
<div align="center">
  <sub>Built with ❤️ for a Healthier World.</sub>
</div>