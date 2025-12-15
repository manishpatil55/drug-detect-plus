
```text
██████╗ ██████╗ ██╗   ██╗ ██████╗    ██████╗ ███████╗████████╗███████╗ ██████╗████████╗
██╔══██╗██╔══██╗██║   ██║██╔════╝    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
██║  ██║██████╔╝██║   ██║██║  ███╗   ██║  ██║█████╗     ██║   █████╗  ██║        ██║   
██║  ██║██╔══██╗██║   ██║██║   ██║   ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   
██████╔╝██║  ██║╚██████╔╝╚██████╔╝   ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   
╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   
                                                           [ SYSTEM_VER: 2.1.0-STABLE ]
```

<div align="center">

> `[STATUS: ONLINE]` • `[PROTOCOL: HYBRID_CORTEX]` • `[UPTIME: 99.9%]`

<a href="https://drug-detect-plus.vercel.app/">
  <img src="https://img.shields.io/badge/INITIATE_SEQUENCE-SUCCESS-00f260?style=for-the-badge&logo=vercel&logoColor=black" />
</a>
<a href="https://github.com/manishpatil55/drug-detect-plus/issues/new?template=bug_report.md">
  <img src="https://img.shields.io/badge/ANOMALY_REPORT-OPEN_CHANNEL-red?style=for-the-badge&logo=github&logoColor=white" />
</a>

</div>

---

## 📟 // SYSTEM_MANIFEST

**Drug Detect AI** is the **Operating System for Medical Intelligence**. 

In 2025, knowledge should not be gated by handwriting or jargon. We built a **Bio-Digital Bridge** that translates physical medicine (atoms) into structured data (bits) instantly.

### Why this exists:
*   **Decentralized Health**: Empowerment starts with knowing what you are taking.
*   **Visual Dominance**: 90% of medical identification happens visually. Text search is legacy tech.
*   **Safety First**: The biggest risk in medication is *interaction error*. We built a logic gate to stop it.

---

## ⚡ // THE_PROCESS (How it Works)

We operate on a **4-Stage Neural Pipeline**. This is what happens in the 800ms after you press "Analyze".

### [STAGE 01]: VISUAL INGESTION
*   **Sensor**: You upload an image (JPG/PNG).
*   **Preprocessing**: The image is converted to a `Base64` stream in-memory.
*   **Cortex**: The `Gemini 2.5 Vision` model scans the pixel data. It performs **OCR (Optical Character Recognition)** to read text on curved bottles, crumpled strips, or low-light blisters.

### [STAGE 02]: SEMANTIC NORMALIZATION
*   **Intent Mapping**: If you type "headache red pill", the vector engine maps this fuzzy query to standard pharmacological terms (e.g., "Naproxen" or "Ibuprofen").
*   **Typo Correction**: `dolo 65` -> `Dolo 650`. The system forgives human error.

### [STAGE 03]: LOGIC CHAIN ANALYSIS
*   This is the **Safety Layer**. If multiple drugs are detected (e.g., *Aspirin* and *Warfarin*):
*   **The Check**: The AI cross-references their chemical interaction matrix.
*   **The Verdict**: It outputs a specific warning: **"High Risk: Anticoagulant Potentiation"**.

### [STAGE 04]: SUPPLY CHAIN ROUTING
*   **Affiliate Engine**: The system identifies the nearest "Digital Dispensaries" (1mg, Apollo, Amazon).
*   **Deep Linking**: It generates a precise, sanitized URL (`search?q=Molecule+Name`) so you land directly on the purchase page.

---

## 🛠️ // ARCHITECTURE_STACK

| [LAYER] | [TECHNOLOGY] | [ROLE] |
| :--- | :--- | :--- |
| **BRAIN** | `Google Gemini 2.5 Flash` | The Cognitive Engine. Low latency, multimodal (Text/Image). |
| **SPINE** | `Python Flask` | Stateless Application Logic. Handles API Rotation & Error Trapping. |
| **FACE** | `Glassmorphism CSS` | High-fidelity UI. Uses backdrop-filters and hardware acceleration. |
| **EDGE** | `Vercel Serverless` | Global Content Delivery Network. |

### Zero-Downtime Protocol (Redundancy)
We implement **Round-Robin Key Rotation**.
*   **Logic**: `List[Keys] = [Key_A, Key_B, Key_C]`
*   **Event**: If `Key_A` hits `429_LIMIT_REACHED`...
*   **Action**: The System auto-switches to `Key_B`.
*   **Result**: Zero downtime for the end-user.

---

## 👾 // PROTOCOL: ANOMALY_REPORTING (Bugs)

If the system exhibits deviant behavior, follow this reporting sequence:

### 1. IDENTIFY THE ANOMALY
*   **TYPE A (Visual Error)**: AI misidentified the drug?
*   **TYPE B (System Error)**: Infinite loading spinner?
*   **TYPE C (Network Error)**: 429/500 Codes?

### 2. OPEN A CHANNEL
*   Go to the **[Issues Tab](https://github.com/manishpatil55/drug-detect-plus/issues)**.
*   Click **New Issue**.

### 3. TRANSMIT LOGS
Structure your report like this for maximum efficiency:
> **[OS]**: iOS 18 / Windows 11
> **[BROWSER]**: Safari / Chrome
> **[ACTION]**: "I uploaded a photo of X..."
> **[RESULT]**: "The spinner never stopped."

*We prioritize Type B and Type C errors.*

---

## 🚀 // DEPLOYMENT_SEQUENCE

### 1. CLONE_REPO
```bash
git clone https://github.com/manishpatil55/drug-detect-plus.git
cd drug-detect-plus
```

### 2. INSTALL_DEPS
```bash
pip install -r requirements.txt
```

### 3. CONFIG_SECRETS
Initiate `.env` file in root directory:
```ini
# [CRITICAL]: Add API Keys (Comma Separated for Redundancy)
GEMINI_API_KEY=key_alpha,key_beta,key_gamma

# [CONFIG]: Model Selection
GEMINI_MODEL=gemini-2.5-flash-lite
```

### 4. IGNITE
```bash
python app.py
```
*System operational at `http://localhost:5000`*

---

## 🖼️ // SYSTEM_PREVIEW

<div align="center">
  <img src="public/drug-detect-ai.png" alt="System Interface Preview" width="100%" style="border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 20px 80px rgba(0,0,0,0.5);" />
  <p style="color: grey; font-size: 0.8rem; margin-top: 10px;">Fig 1.0: The Neural Interface in Action</p>
</div>

---

## 👨‍💻 // OPERATOR_LOG

**Command**: [Manish Patil](https://github.com/manishpatil55)  
**Comms**: [manishpatil55@icloud.com](mailto:manishpatil55@icloud.com)  

**License**: `CC BY-NC 4.0` (Open Source / Restricted Commercial).

<br />
<div align="center">
  <code>[END_OF_TRANSMISSION]</code>
</div>
