# Eye Recognition System 👁️

A real-time eye recognition application built with **MediaPipe**, **InsightFace**, and **ChromaDB**. This system allows you to enroll individuals and identify them in real-time with high precision using GPU acceleration.

## ✨ Features
- **Neural & Color Hybrid**: Combines ArcFace embeddings with iris color analysis for maximum accuracy.
- **DirectML Support**: High-performance GPU acceleration on Windows (NVIDIA, AMD, Intel).
- **Temporal Voting**: Multi-frame consensus logic to eliminate false positives and flickering.
- **Image Enhancement**: CLAHE-based eye preprocessing for better texture recognition.
- **Dual Enrollment**: Support for both live webcam and static image registration.

---

## 🛠️ Requirements
- Python 3.10+
- Webcam
- Windows GPU with DirectX 12 support (for DirectML)

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/eye-analyser.git
cd eye-analyser
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Step 1: Enrollment
You can enroll users in two ways:

**A. Via Webcam (Recommended for accuracy):**
```bash
python enroll.py
```
- The system will ask if you want to clear the database.
- Follow the on-screen instructions (look straight, eyes open).
- Press **SPACE** to capture 20 high-quality samples.

**B. Via Static Image:**
```bash
python enroll_image.py "path/to/image.jpg" "UserName"
```

### Step 2: Real-Time Recognition
Run the main recognition loop:
```bash
python main.py
```
- **GREEN OVERLAY**: User recognized.
- **RED OVERLAY**: Unknown user or low confidence.
- Results are smoothed over 10 frames for stability.

---

## ⚙️ Project Structure
- `eye_processor.py`: MediaPipe landmarks, CLAHE enhancement, and eye cropping.
- `model_handler.py`: InsightFace inference with Color Feature extraction and DirectML setup.
- `db_handler.py`: ChromaDB vector search management.
- `enroll.py`: Interactive live enrollment script.
- `enroll_image.py`: Script for bulk enrollment from photos.
- `main.py`: Main application loop with temporal voting logic.

---

## ⚖️ License
MIT License. Feel free to use and modify.
