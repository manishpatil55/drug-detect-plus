
<div align="center">
  <a href="https://drug-detect-plus.vercel.app/">
    <img src="public/drug-detect-ai.jpeg" alt="Drug Detect AI Hero" width="100%" style="border-radius: 24px; box-shadow: 0 0 50px rgba(0, 255, 200, 0.15); border: 1px solid rgba(255,255,255,0.08);" />
  </a>

  <br />
  <br />

  <h1 style="font-size: 4rem; font-weight: 900; letter-spacing: -3px; background: -webkit-linear-gradient(135deg, #00F5A0 0%, #00D9F5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(0 0 40px rgba(0,255,240,0.3));">
    Drug Detect AI ⚡️
  </h1>

  <p style="font-size: 1.4rem; font-weight: 500; color: #b0c4de; max-width: 650px; margin: 0 auto 35px; line-height: 1.6;">
    The <strong>Medical Singularity Engine</strong>. <br/>
    Bridging the gap between <strong>Analog Biology</strong> and <strong>Digital Intelligence</strong>.
  </p>

  <div style="display: flex; gap: 15px; justify-content: center; align-items: center; flex-wrap: wrap;">
    <a href="https://drug-detect-plus.vercel.app/" style="background: rgba(255,255,255,0.1); backdrop-filter: blur(12px); padding: 14px 28px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.2); color: white; text-decoration: none; font-weight: 700; transition: transform 0.2s;">🔴 Launch Neural Interface</a>
    <a href="https://github.com/manishpatil55/drug-detect-plus/issues" style="background: rgba(255,255,255,0.03); padding: 14px 28px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.05); color: #ccc; text-decoration: none; font-weight: 600;">🐛 Report Anomaly</a>
  </div>

  <br />

  <p align="center" style="opacity: 0.85;">
    <img src="https://img.shields.io/badge/System-Operational-success?style=for-the-badge&logo=vercel&logoColor=white&color=000000" />
    <img src="https://img.shields.io/badge/Vision_Model-Gemini_2.5-blue?style=for-the-badge&logo=google&logoColor=white&color=0066cc" />
    <img src="https://img.shields.io/badge/Protocol-Stateless_Flask-lightgrey?style=for-the-badge&logo=flask&logoColor=white&color=333333" />
    <img src="https://img.shields.io/github/stars/manishpatil55/drug-detect-plus?style=for-the-badge&logo=github&color=181717" />
  </p>
</div>

<br />

> **⚠️ Cyber-Safety Protocol**: This tool utilizes Probabilistic AI. Results are derived from visual pattern matching and are for **informational purposes only**. Do not ingest substances based solely on machine output. Consult a biological healthcare provider.

---

## 🌌 The Architecture of Health

**Drug Detect AI** is not just an app; it is a specialized **search engine for physical reality**. 

In an era of generic LLMs, this system is fine-tuned for **Pharmacological Recognition**. It bypasses the need for textual input, allowing users to query the database using raw photons (images) of strips, bottles, or blisters.

### 🧬 Core Capabilities

| Module | Function | Technology |
| :--- | :--- | :--- |
| **Visual Cortex** | Decodes crumpled foils, handwritten text, and pill shapes. | `Gemini 2.5 Vision` |
| **Semantic Map** | Translates fuzzy queries ("red headache pill") into chemical names. | `Vector Similarity` |
| **Safety Net** | Cross-references multiple drugs to detect hazardous interactions. | `Logic Chain Analysis` |
| **Supply Link** | Locates the nearest digital dispensary (1mg, Apollo, Amazon). | `Dynamic Routing` |

---

## ⚡️ Deep Technical Breakdown

### 1. Hybrid Cortex Engine
The system employs a dual-pathway processing unit:
*   **Path A (Image)**: Images are converted to base64 types and fed into the `gemini-2.5-flash-lite` model. The model performs OCR (Optical Character Recognition) + Object Detection simultaneously to infer the brand and salt composition.
*   **Path B (Text)**: Text queries undergo "Intent Normalization". A query like "dolo 65" is auto-corrected to "Dolo 650" before processing.

### 2. Zero-Downtime Reliability Protocol
We implement a **Round-Robin Key Rotation** strategy at the application layer.
*   **The Problem**: Free-tier AI models have rate limits (RPM/RPD).
*   **The Solution**: The backend maintains a `List[API_KEY]`. If a request fails with `429 Resource Exhausted`, the exception handler catches the specific error signature, rotates the active key index, and retries the request transparently within milliseconds.

### 3. Stateless Privacy Architecture
**Privacy by Design**.
*   **No Database**: We do not store your medical queries.
*   **No Image Retention**: Images are processed in RAM and discarded immediately after analysis.
*   **Ephemeral Sessions**: Everything happens in the "Now". Your health data belongs to you, not a server.

---

## 📂 System File Hierarchy

Understanding the codebase structure for contributors:

```bash
drug-detect-plus/
├── app.py                 # [CORE] The Flask Neural Backbone. Handles Routes & AI Logic.
├── requirements.txt       # [DEPS] The dependency manifest (Flask, Google-GenAI).
├── .env                   # [SECRETS] API Keys & Model Configuration (GitIgnored).
├── procfile               # [DEPLOY] Gunicorn entry point for platform deployment.
├── public/                # [ASSETS] Static logos, favicons, and social previews.
├── templates/
│   └── index.html         # [INTERFACE] The Single-Page Application (SPA). 
│                          # Contains all CSS (Glassmorphism), HTML, and JS logic.
└── README.md              # [DOCS] You are reading this node.
```

---

## 🎯 Use Cases (Target Personas)

*   **👵 The Elderly**: Can't read small print on medicine strips? Just take a photo.
*   **✈️ Travelers**: In a foreign country with foreign medicine brands? Identify them instantly.
*   **💊 Pharmacists**: Quickly check drug interactions for customers carrying multiple prescriptions.
*   **🧪 Students**: Visualize mechanism of action and dosage for study purposes.

---

## 🚀 Initialization Sequence (Deploy)

### Phase 1: Clone
```bash
git clone https://github.com/manishpatil55/drug-detect-plus.git
cd drug-detect-plus
```

### Phase 2: Inject Dependencies
```bash
pip install -r requirements.txt
```

### Phase 3: Environment Config
Create a `.env` file in the root. This is your authentication layer.
```ini
# ROTATION POOL: Add multiple keys separated by commas for infinite uptime.
GEMINI_API_KEY=key_alpha,key_beta,key_gamma

# MODEL SELECTOR: Optimized for speed.
GEMINI_MODEL=gemini-2.5-flash-lite
```

### Phase 4: Ignite
```bash
python app.py
```
*Access the interface at `http://localhost:5000`*

---

## 📡 Troubleshooting Matrix

| Error Code | Meaning | Solution |
| :--- | :--- | :--- |
| **429** | Resource Exhausted | Add more keys to `.env`. The system will auto-rotate. |
| **500** | Internal Error | Check console logs. Usually invalid Image format. |
| **Infinite Spin** | JS/Backend disconnect | Refresh. Check internet connection. |

---

## 📜 Intellectual Property

**Architect**: [Manish Patil](https://github.com/manishpatil55)  
**Contact**: [manishpatil55@icloud.com](mailto:manishpatil55@icloud.com)  

**License**: [CC BY-NC 4.0](LICENSE).
*   **Open Source**: Yes.
*   **Commercial Use**: Restricted.
*   **Modification**: Allowed with Attribution.

<br />
<div align="center">
  <sub>Engineered for the Bio-Digital Future. 🧬</sub>
</div>
