# Eye Recognition & Model Comparison Framework 👁️⚖️

A professional framework for real-time periocular (eye area) recognition and comparison between different deep learning architectures. This project compares a specialized Face Recognition model (**ArcFace**) with a general-purpose Vision Transformer (**DINOv2**).

## 📊 Project Overview
The goal of this project is to evaluate how different neural network architectures perform when restricted only to the eye area. It includes real-time processing, vector storage, and a comparison engine with automated visualization.

### ⚔️ The Contenders
- **V1: ArcFace (InsightFace Buffalo_L)**
  - **Type**: CNN-based Metric Learning (Specialized for humans).
  - **Focus**: Global facial geometry and identity-invariant features.
  - **Strength**: High stability, low sensitivity to lighting and minor rotations.
- **V2: DINOv2 (Meta ViT-S/14)**
  - **Type**: Vision Transformer (Self-supervised Foundation Model).
  - **Focus**: Fine-grained local textures and patch-level details.
  - **Strength**: Extremely high sensitivity to iris patterns and skin texture.

---

## ✨ Key Features
- **Hybrid Embedding**: Neural vectors augmented with Iris Color features for better precision.
- **DirectML Acceleration**: Native Windows GPU support (NVIDIA, AMD, Intel).
- **Battle Mode**: Simultaneous real-time inference from both models on a single video stream.
- **Auto-Visualization**: Generates comparison charts (`.png`) from session logs (`.csv`).
- **Temporal Voting**: Multi-frame consensus logic to eliminate recognition flickering.

---

## 🚀 Getting Started

### 1. Installation
```bash
# Install core dependencies
pip install -r v1_arcface/requirements.txt

# Install PyTorch (Required for DINOv2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Plotting tools
pip install pandas matplotlib
```

### 2. Enrollment (Data Collection)
You must enroll your eyes in both databases for comparison:
```bash
# Enroll in V1
cd v1_arcface
python enroll.py
cd ..

# Enroll in V2
cd v2_dinov2
python enroll.py
cd ..
```

### 3. Running the Comparison (Battle Mode)
Run both models side-by-side:
```bash
python compare.py
```
*Press **'q'** to exit and save the log.*

### 4. Generating the Report
Create a comparison chart from your session data:
```bash
python plot_results.py
```
This will generate `recognition_comparison.png`.

## 🐳 Docker Deployment

The project is fully containerized with GPU support. You can run different versions using **Docker Compose**:

**Build the environment:**
```bash
docker-compose build
```

**Run specific services:**
```bash
# Run the Comparison Battle
docker-compose run compare

# Run only ArcFace (V1)
docker-compose run arcface

# Run only DINOv2 (V2)
docker-compose run dinov2
```

> **Note:** For webcam access inside Docker on Windows/Mac, additional configuration (like WSL2 USB pass-through) may be required.

---

## 📂 Project Structure
- `/v1_arcface`: ArcFace-based implementation (Robust & Fast).
- `/v2_dinov2`: DINOv2-based implementation (Texture-sensitive).
- `compare.py`: The "Battle" engine running both models simultaneously.
- `plot_results.py`: Visualization script for performance analysis.
- `comparison_log.csv`: Raw distance data from your last session.

---

## 📈 Analysis Results
Based on our tests, **ArcFace (V1)** provides more stable identity confirmation for "Access Control" scenarios, while **DINOv2 (V2)** is more suitable for "Liveness Detection" or "Fine-grained Detail" analysis due to its high sensitivity to local changes.

---

## ⚖️ License
MIT License. Created for research and educational purposes.
