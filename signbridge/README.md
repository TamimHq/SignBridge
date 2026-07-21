# SignBridge — Bidirectional Sign Language Translation

> Breaking communication barriers between the Deaf and hearing communities through AI — supporting **Bangladeshi Sign Language (BdSL ↔ Bengali)** and **American Sign Language (ASL ↔ English)**.

![Status](https://img.shields.io/badge/status-working%20prototype-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Flutter](https://img.shields.io/badge/Frontend-Flutter-02569B)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

**SignBridge** is a bidirectional sign language communication system: it recognizes signs from camera input and generates sign animations from typed text, letting a Deaf signer and a hearing person hold a conversation through one app.

Most sign language research targets high-resource languages like ASL. SignBridge focuses primarily on **Bangladeshi Sign Language**, a critically under-resourced language with almost no annotated datasets or pretrained models — serving the 2.6+ million Deaf and hard-of-hearing people in Bangladesh.

## Features

| Direction | BdSL ↔ Bengali | ASL ↔ English |
|---|---|---|
| Sign → Text (camera) | ✅ 200 signs, 72.7% accuracy | ❌ not viable (see Findings) |
| Text → Sign (avatar) | ✅ 200 signs | ✅ 102 words |
| Text → Speech | ✅ | ✅ |
| Speech → Text | ✅ Whisper | ✅ Whisper |

- **Pose-based recognition** — MediaPipe hand and upper-body landmarks, 144-dim per frame
- **Skeleton avatar** — animated sign output rendered from keypoints (ASL) and OpenPose frames (BdSL)
- **Cross-platform app** — Flutter for Android, iOS, and web
- **Modular backend** — FastAPI service for recognition and transcription

## Architecture

```
┌──────────────────┐   video clip    ┌──────────────────┐
│   Flutter App    │ ──────────────▶ │ FastAPI Backend  │
│  camera / text   │ ◀────────────── │                  │
└──────────────────┘   word + conf   └────────┬─────────┘
        │                                     │
        │ text → lookup engine                ▼
        ▼                          ┌─────────────────────┐
┌──────────────────┐               │ MediaPipe Tasks     │
│ Skeleton Avatar  │               │ pose + hands → 144d │
└──────────────────┘               └──────────┬──────────┘
                                              ▼
                                   ┌─────────────────────┐
                                   │ Bi-LSTM classifier  │
                                   │ 30×144 → gloss      │
                                   └─────────────────────┘
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Flutter, Dart, Provider |
| Backend | FastAPI, Uvicorn, Python 3.13 |
| Computer Vision | MediaPipe Tasks (PoseLandmarker, HandLandmarker), OpenCV |
| ML | PyTorch (Bi-LSTM), NumPy |
| Speech | flutter_tts, OpenAI Whisper |
| Datasets | SignBD-Word (BdSL), How2Sign (ASL) |

## Technical Approach

1. **Keypoint extraction** — Each frame yields a 144-dim vector: 6 upper-body joints (shoulders, elbows, wrists) + 21 left-hand + 21 right-hand landmarks, each as (x, y, confidence). Clips are resampled to 30 frames and normalized by centering on the mid-shoulder and scaling by shoulder width, making features invariant to camera distance and position.
2. **Recognition** — A 2-layer bidirectional LSTM (hidden 160) with mean-pooling over time classifies the 30×144 sequence into one of 200 glosses. Trained with mirroring, scaling, time-warp, and jitter augmentation to offset limited samples per class.
3. **Generation** — Typed text is tokenized, reordered toward sign gloss order, and matched against a vocabulary index; each matched word plays its recorded pose sequence in the avatar, with fingerspelling and skip fallbacks for out-of-vocabulary words.

## Results

**BdSL recognition — 72.7% test accuracy across 200 classes** (chance = 0.5%), trained on 4,800 clips from 16 signers in two camera framings. Performance is highly uneven: many signs reach 100%, while a few reach 0% — those depend on facial expression or fine finger detail that upper-body and hand keypoints don't capture.

**ASL recognition — not viable from How2Sign.** Despite 35,000 available clips, only ~700 are single-word, they average 3-7 frames, and they are cut from continuous signing so each contains co-articulation from neighbouring signs. The best model reached 57.3% on four classes. ASL is therefore supported for text→sign only.

**Finding:** dataset *structure* mattered more than dataset *size*. SignBD-Word's 4,800 isolated-word clips trained a usable 200-class recognizer; How2Sign's 35,000 continuous clips could not train a reliable 4-class one. For low-resource sign languages, this argues for collecting isolated-word data rather than pursuing scale.

## Getting Started

### Backend

```bash
cd server
pip install -r requirements.txt
python server.py            # serves on 0.0.0.0:8000
```

Place the trained model in `server/models/`. MediaPipe task bundles download automatically on first run.

To reach the server from a phone, use your machine's LAN IP (`ipconfig` / `ifconfig`) and make sure both devices are on the same network:

```
http://<your-ip>:8000/health
```

### Frontend

```bash
cd signbridge
flutter pub get
flutter run
```

Set the server URL in `lib/core/api_client.dart` to your machine's IP (not `localhost`, which on a phone means the phone itself).

BdSL avatar frames are not committed due to size — regenerate them from the dataset with `copy_bdsl_assets.py`, then `flatten_bdsl_assets.py`.

## Roadmap

- [ ] Add facial keypoints to recover signs that currently fail
- [ ] Retrain ASL recognition on WLASL (isolated-word ASL dataset)
- [ ] Expand BdSL vocabulary beyond 200 signs
- [ ] On-device inference via TensorFlow Lite / ONNX
- [ ] Hosted demo deployment

## Datasets

This project uses two publicly available datasets. Please cite them in any academic work.

**SignBD-Word (BdSL)**

> Ataher Sams (2022). *SignBD-Word: Video-Based Bangla Word-Level Sign Language Dataset.* 2023 14th International Conference on Computing Communication and Networking Technologies (ICCCNT), Zenodo. https://doi.org/10.5281/zenodo.6779843

**How2Sign / How2 (ASL)**

```bibtex
@inproceedings{sanabria18how2,
  title = {{How2:} A Large-scale Dataset For Multimodal Language Understanding},
  author = {Sanabria, Ramon and Caglayan, Ozan and Palaskar, Shruti and Elliott, Desmond and Barrault, Lo\"ic and Specia, Lucia and Metze, Florian},
  booktitle = {Proceedings of the Workshop on Visually Grounded Interaction and Language (ViGIL)},
  year = {2018},
  organization = {NeurIPS},
  url = {http://arxiv.org/abs/1811.00347}
}
```

The trained BdSL model in this repository is derived from SignBD-Word and is subject to that dataset's licence terms. Datasets themselves are not redistributed here — download them from the original sources.

## Author

**Md Tamim Haque**
[Portfolio](https://tamimportfolio.com) · [LinkedIn](https://linkedin.com/in/tamimhaque) · [GitHub](https://github.com/TamimHq)

## License

MIT — see [LICENSE](LICENSE). Note that dataset and derived-model terms apply separately, as described above.