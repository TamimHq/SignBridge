# SignBridge — Bidirectional Sign Language Translation

> Breaking communication barriers between the Deaf and hearing communities through AI — supporting both **American Sign Language (ASL ↔ English)** and **Bangladeshi Sign Language (BdSL ↔ Bengali)**.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Flutter](https://img.shields.io/badge/Frontend-Flutter-02569B)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Overview

**SignBridge** is a bidirectional sign language communication system designed to translate between sign language and spoken/written language in **both directions** — recognizing signs from video input and generating sign animations from text.

Unlike most existing systems that focus solely on high-resource sign languages like ASL, SignBridge also targets **Bangladeshi Sign Language (BdSL)** — a critically under-resourced language with almost no annotated datasets or pretrained models. This makes SignBridge both a technical system and a step toward **digital equity** for the 2.6+ million Deaf and hard-of-hearing individuals in Bangladesh.

## 🎯 Key Features

- **Bidirectional translation** — sign-to-text recognition *and* text-to-sign generation
- **Multilingual pipelines** — ASL ↔ English and BdSL ↔ Bengali
- **Pose-based recognition** — keypoint extraction using MediaPipe for hand, body, and facial landmarks
- **Skeleton avatar generation** — animated skeletal output for text-to-sign direction
- **Cross-platform app** — Flutter frontend for Android, iOS, and web
- **Modular API backend** — FastAPI service exposing recognition and generation endpoints

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Flutter App    │────▶│   FastAPI Backend │────▶│  Recognition Model   │
│  (camera/text)  │◀────│   (REST API)      │◀────│  (pose → gloss → text)│
└─────────────────┘      └──────────────────┘      └─────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────────┐
                        │  Keypoint Pipeline    │
                        │  (MediaPipe → JSON)   │
                        └──────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Flutter, Dart |
| **Backend** | FastAPI, Python |
| **Computer Vision** | MediaPipe, OpenCV |
| **ML / DL** | TensorFlow / PyTorch <!-- ⚠️ VERIFY which one you used -->, NumPy |
| **Data Format** | NumPy-to-JSON keypoint serialization |
| **Datasets** | How2Sign (ASL), SignBD-Word (BdSL) <!-- ⚠️ VERIFY dataset names --> |

## 🔬 Technical Approach

1. **Keypoint Extraction** — Video frames are processed through MediaPipe to extract hand, pose, and facial keypoints, converted into structured JSON sequences.
2. **Sign Recognition** — Keypoint sequences are modeled to classify/translate signs into glosses and then natural language.
3. **Sign Generation** — Text input is mapped to a sequence of poses, rendered as a skeleton-based avatar animation.

## 🚧 Project Status

SignBridge is an **active, in-progress research project**. Currently implemented:
- ✅ Keypoint extraction and NumPy-to-JSON pipeline
- ✅ FastAPI backend scaffold with recognition endpoints
- ✅ Flutter app foundation
- 🔄 Skeleton avatar animation (in progress)
- 🔄 BdSL model training and evaluation (in progress)

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/TamimHq/SignBridge.git
cd SignBridge

# Backend setup
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend setup
cd ../frontend
flutter pub get
flutter run
```
<!-- ⚠️ VERIFY folder names match your actual repo structure -->

## 🗺️ Roadmap

- [ ] Expand BdSL vocabulary coverage
- [ ] Improve recognition accuracy with transfer learning from ASL
- [ ] Refine skeleton avatar smoothness
- [ ] Deploy hosted demo

## 👤 Author

**Md Tamim Haque**
Machine Learning Researcher & Developer
[Portfolio](https://tamimportfolio.com) · [LinkedIn](https://linkedin.com/in/tamimhaque) · [GitHub](https://github.com/TamimHq)

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
