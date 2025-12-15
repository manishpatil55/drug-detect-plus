<div align="center">
  <a href="https://drug-detect-plus.vercel.app/">
    <img src="public/drug-detect-ai.jpeg" alt="Drug Detect AI - Smart Medicine Analyzer" width="100%" style="border-radius: 16px; box-shadow: 0 8px 30px rgba(0, 150, 150, 0.3); border: 1px solid rgba(255,255,255,0.1);" />
  </a>

  <br />
  <br />

  <h1>💊 Drug Detect AI</h1>
  <p class="lead"><strong>The Next-Gen Medical Intelligence Engine.</strong></p>

  <p>
    <a href="https://drug-detect-plus.vercel.app/"><strong>🔴 View Live Demo</strong></a> • 
    <a href="https://github.com/manishpatil55/drug-detect-plus/issues"><strong>🐛 Report Bug</strong></a> • 
    <a href="#-license"><strong>📜 License</strong></a>
  </p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Status-Production-success?style=for-the-badge&logo=vercel" />
    <img src="https://img.shields.io/badge/AI-Gemini%202.5%20Flash-blue?style=for-the-badge&logo=google" />
    <img src="https://img.shields.io/badge/Stack-Flask%20%2B%20Python-yellow?style=for-the-badge&logo=python" />
    <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-orange?style=for-the-badge" />
  </p>
</div>

---

## � Table of Contents
- [✨ Features](#-features)
- [🏗️ System Architecture](#-system-architecture)
- [📦 Installation & Setup](#-installation--setup)
- [🛡️ Security & Reliability](#-security--reliability)
- [💸 Monetization (Affiliates)](#-monetization-affiliates)
- [📜 License](#-license)
- [📞 Contact](#-contact)

---

## ✨ Features

**Drug Detect AI** is more than just an image scanner. It is a comprehensive *Medical Search Engine* designed to provide instant, actionable health data.

### 🔍 Hybrid Search Engine (Text + Vision)
- **Visual Intelligence**: Upload extensive formats (`JPEG`, `PNG`, `WEBP`). The system extracts text, shape, and color to identify medicines with 99% accuracy using **Google Gemini 1.5/2.5 Pro**.
- **Fuzzy Text Search**: Type "Dolo", "Dolo 650", or even "med for headache". The NLP engine handles typos, synonyms, and descriptive queries.

### 🧠 Deep Medical Insights
For every detected medicine, you get a structured breakdown:
- **Usage**: What is it for?
- **Dosage**: Adult vs. Pediatric guidelines.
- **Mechanism**: How does it work?
- **Side Effects**: Common vs. Rare warnings.
- **Habit Forming**: Addiction risk assessment.

### ⚡ Performance & UX
- **Glassmorphism Design**: A premium, "frosted glass" UI with ambient particle systems.
- **Mobile First**: Fully responsive layout that adapts to any viewport.
- **Instant Translation**: Built-in support for multi-language responses (Hindi, Spanish, French, etc).

---

## 🏗️ System Architecture

The application checks for **API Limits** and **Availability** in real-time.

```mermaid
graph TD
    User[User] -->|Uploads Image/Text| FE[Frontend (HTML/JS)]
    FE -->|POST Request| BE[Flask Backend]
    
    subgraph "reliability Layer"
    BE -->|Try Key 1| G1[Gemini AI]
    G1 -->|Success| Response
    G1 -->|429 Error| Rotation[Key Rotation Logic]
    Rotation -->|Try Key 2| G2[Gemini AI]
    Rotation -->|Try Key 3| G3[Gemini AI]
    end
    
    Response -->|JSON Data| FE
    FE -->|Render| UI[User Interface]
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- A Google Cloud Project with **Gemini API** enabled.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/manishpatil55/drug-detect-plus.git
cd drug-detect-plus
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a `.env` file in the root directory. This is critical for the app to function.

```env
# CRITICAL: API Keys
# You can add multiple keys separated by commas for auto-failover.
GEMINI_API_KEY=AIzaSy...Key1,AIzaSy...Key2,AIzaSy...Key3

# OPTIONAL: Model Configuration
# Defaults to "gemini-2.5-flash-lite" if not set.
GEMINI_MODEL=gemini-2.5-flash-lite
```

### 4️⃣ Run the Application
```bash
python app.py
```
Access the dashboard at `http://localhost:5000`.

---

## 🛡️ Security & Reliability

### 🔄 Multi-Key Rotation System
The biggest challenge with free-tier AI APIs is the **Rate Limit (RPD/RPM)**.
Drug Detect AI solves this with an **Enterprise-Grade Rotation System**:
1.  **Detection**: It listens for `429 Resource Exhausted` or `Quota Exceeded` errors.
2.  **Rotation**: If Key 1 fails, it instantly switches to Key 2, then Key 3.
3.  **Seamless**: The user never sees an error page. The request simply takes 0.5s longer.

### 🔒 Privacy by Design
- **No Database**: We do not store user images or search queries.
- **Ephemeral Storage**: Images are processed in RAM and discarded immediately after analysis.

---

## 💸 Monetization (Affiliates)

This project is pre-configured to generate revenue via Affiliate Marketing.

- **Dynamic Link Generation**: The AI automatically creates "Buy Now" links for:
    - [1mg](https://www.1mg.com/)
    - [Apollo Pharmacy](https://www.apollopharmacy.in/)
    - [Netmeds](https://www.netmeds.com/)
    - [Amazon](https://www.amazon.in/)
- **Amazon Affiliate Tag**: The code specifically injects `&tag=drugdetectai-21` into Amazon links.
    - *To claim this revenue, replace this tag with your own Amazon Associate ID in `app.py`.*

---

## 📜 License

**Copyright © 2025 Manish Patil.**

This project is licensed under the **[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)** (CC BY-NC 4.0).

| ✅ You CAN | ❌ You CANNOT |
| :--- | :--- |
| **Share**: Copy and redistribute the material. | **Sell**: Use for commercial purposes without permission. |
| **Adapt**: Remix, transform, and build upon it. | **Hide**: Distribute without credit (Attribution). |

> *If you wish to use this software for a commercial SaaS, startup, or paid service, please contact the developer for a specialized license.*

---

## 📞 Contact

**Manish Patil**  
*Full Stack Developer & AI Enthusiast*

- 🌐 **GitHub**: [@manishpatil55](https://github.com/manishpatil55)
- 📧 **Email**: [manishpatil55@icloud.com](mailto:manishpatil55@icloud.com)

---

<div align="center">
  <sub>Built with ❤️ and ☕ in 2025.</sub>
</div>