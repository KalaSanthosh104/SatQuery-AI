# SatQuery-AI

## Vision-Language Assistant for Remote Sensing

SatQuery-AI is a remote-sensing Vision-Language system developed for **Smart India Hackathon 2026 – Problem Statement SIH26167**.

The project combines satellite imagery with Vision-Language Models (VLMs) to support image understanding, question answering, and change analysis from remote-sensing data.

---

## 🚀 Key Capabilities

### 1. Single Image Analysis — RSVQA

The RSVQA component performs Visual Question Answering on remote-sensing images.

It combines:

- Vision Transformer (ViT)
- BERT-based language encoding
- Multimodal feature fusion
- Task-specific prediction heads

The model supports questions related to:

- Object presence
- Composition and comparison
- Rural / Urban classification
- Object counting

📁 Implementation:

`RSVQA/`

---

### 2. Bi-Temporal Change Analysis — CDVQA

The CDVQA component analyzes two images acquired at different times and answers questions about changes between them.

It is designed for questions involving:

- Change / no-change
- Changed objects
- Buildings
- Surface changes
- Other change-related categories

📁 Implementation:

`CDVQA/`

The trained checkpoint is provided separately as:

`best_cdvqa_model.pth.zip`

---

### 3. Optical + SAR Analysis

The Optical + SAR component combines:

- Sentinel-1 SAR imagery
- Sentinel-2 optical imagery

The architecture uses:

**CROMA → Projector → Qwen**

The system combines multimodal satellite information before generating answers to remote-sensing questions.

📁 Implementation:

`Optical-SAR-VQA/`

---

# 🧠 System Architecture

```text
                    SatQuery-AI
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      Single Image   Bi-Temporal    Optical + SAR
        Analysis      Analysis        Analysis
          │              │              │
          ▼              ▼              ▼
        RSVQA           CDVQA       CROMA + Qwen
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Remote-Sensing
                 Question Answering