# 🛰️ SatQuery-AI

## Vision-Language Assistant for Remote Sensing

SatQuery-AI is a **remote-sensing Vision-Language system** developed for **Smart India Hackathon 2026 – Problem Statement SIH26167**.

The project combines satellite imagery, computer vision, deep learning, and Vision-Language Models (VLMs) to enable intelligent understanding and natural-language querying of remote-sensing data.

---

## 🎯 Problem Statement

Remote-sensing imagery contains large amounts of information about the Earth's surface, but extracting useful information often requires specialized knowledge and manual analysis.

SatQuery-AI aims to simplify this process by allowing users to interact with satellite imagery using **natural-language questions**.

The system supports three major analysis scenarios:

- 🛰️ Single Image Analysis
- 🔄 Bi-Temporal Change Analysis
- 🌍 Optical + SAR Multimodal Analysis

---

## 🚀 What is SatQuery-AI?

SatQuery-AI provides a unified framework for asking questions about satellite imagery.

Instead of manually inspecting large remote-sensing images, users can provide an image or a pair of images along with a question.

The corresponding AI model processes the imagery and generates an answer.

### Overall Concept

```text
             Satellite Imagery
                    │
                    ▼
          ┌───────────────────┐
          │    SatQuery-AI    │
          └─────────┬─────────┘
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
        RSVQA     CDVQA   Optical + SAR
          │         │         │
          ▼         ▼         ▼
       Single     Change    Multimodal
        Image    Analysis    Analysis
          │         │         │
          └─────────┼─────────┘
                    ▼
             Natural Language
                  Answer
```

---

## ✨ Key Features

- 🛰️ Remote-sensing image understanding
- 💬 Natural-language question answering
- 🧠 Vision-Language Models
- 🔄 Bi-temporal change analysis
- 🌍 Optical + SAR multimodal analysis
- 📊 Support for multiple remote-sensing datasets
- 🤖 Multiple specialized AI models
- 🗺️ Satellite-image based visual reasoning

---

# 🧠 AI Modules

SatQuery-AI consists of three major AI components.

| Module | Input | Purpose | Architecture |
|---|---|---|---|
| **RSVQA** | Single optical image + question | Remote-sensing Visual Question Answering | ViT + BERT + Multimodal Fusion |
| **CDVQA** | Two temporal images + question | Bi-temporal change analysis | Change Analysis + VQA |
| **Optical + SAR VQA** | Sentinel-1 + Sentinel-2 + question | Multimodal satellite analysis | CROMA + Projector + Qwen |

---

# 1️⃣ Single Image Analysis — RSVQA

The **RSVQA** component performs Visual Question Answering on remote-sensing images.

It combines:

- Vision Transformer (ViT)
- BERT-based language encoding
- Shared multimodal feature fusion
- Task-specific prediction heads

The model supports questions related to:

- Object presence
- Composition and comparison
- Rural / Urban classification
- Object counting

### Architecture

```text
Remote-Sensing Image
        │
        ▼
Vision Transformer
        │
        ▼
Visual Features
        │
        ├──────────────┐
        │              │
        ▼              ▼
    Question      BERT Encoder
        │              │
        └──────┬───────┘
               ▼
       Multimodal Fusion
               │
               ▼
       Task-Specific Heads
               │
               ▼
             Answer
```

### Implementation

```text
RSVQA/
```

Detailed documentation:

[RSVQA Documentation](RSVQA/README.md)

---

# 2️⃣ Bi-Temporal Change Analysis — CDVQA

The **CDVQA** component analyzes two images acquired at different times and answers questions about changes between them.

The system is designed for questions involving:

- Change / No Change
- Changed objects
- Buildings
- Surface changes
- Other change-related categories

### Architecture

```text
Image at Time T1 ──┐
                   │
                   ▼
             Change Analysis
                   │
                   │
Image at Time T2 ──┘
                   │
                   ▼
               VQA Model
                   │
                   ▼
                 Answer
```

### Implementation

```text
CDVQA/
```

The trained CDVQA checkpoint is maintained separately from the source-code repository.

---

# 3️⃣ Optical + SAR VQA

The Optical + SAR module combines information from two satellite modalities.

### Sentinel-1

Provides **Synthetic Aperture Radar (SAR)** information.

### Sentinel-2

Provides **optical multispectral imagery**.

The architecture uses:

```text
CROMA → Projector → Qwen
```

### Architecture

```text
Sentinel-1 SAR ──────┐
                     │
                     ▼
                   CROMA
                     │
Sentinel-2 Optical ──┘
                     │
                     ▼
                 Projector
                     │
                     ▼
                   Qwen
                     │
                     ▼
                  Answer
```

### Implementation

```text
Optical-SAR-VQA/
```

The Qwen-related files are stored under:

```text
Optical-SAR-VQA/qwen/
```

Detailed documentation:

[Optical + SAR Documentation](Optical-SAR-VQA/README.md)

---

# 🛰️ Remote-Sensing Modalities

SatQuery-AI works with multiple types of satellite information.

| Analysis | Input |
|---|---|
| Single Image VQA | Remote-sensing optical imagery |
| Bi-Temporal Change Analysis | Images from two different time periods |
| Optical + SAR VQA | Sentinel-2 optical + Sentinel-1 SAR |

---

# 📊 Datasets

## RSVQA-LR

The RSVQA implementation uses the **RSVQA-LR** dataset.

The documented dataset contains:

- 772 TIFF images
- 57,223 training samples
- 10,005 validation samples
- 10,004 official test samples

The questions cover:

- Presence
- Composition
- Rural / Urban classification
- Counting

More details are available in the [RSVQA documentation](RSVQA/README.md).

---

## CDVQA

CDVQA (**Change Detection Visual Question Answering**) is used for **bi-temporal change analysis**.

The model receives two images of the same geographic area captured at different times and a natural-language question about the changes between them.

### Dataset Statistics

| Property | Value |
|---|---:|
| Unique Image Pairs | **1,600** |
| Training Image Records | **25,600** |
| Questions | **65,967** |
| Answers | **65,967** |
| Unique Answers | **19** |

The dataset contains questions related to different types of changes between the two images.

### Question Categories

The CDVQA implementation supports questions such as:

- Change / No Change
- Increase / No Increase
- Decrease / No Decrease
- Smallest Change
- Largest Change
- Change to What
- Change Ratio
- Change Ratio Types

The model supports **8 question types**:

```text
change_or_not
increase_or_not
decrease_or_not
smallest_change
largest_change
change_to_what
change_ratio
change_ratio_types

---

## Optical + SAR VQA

The Optical + SAR implementation uses paired:

- Sentinel-1 SAR
- Sentinel-2 optical

imagery.

The documented VQA dataset contains:

- 2,500 question-answer records
- 2,000 training records
- 250 validation records
- 250 test records

More details are available in the [Optical + SAR documentation](Optical-SAR-VQA/README.md).

---

# 🏗️ Overall System Architecture

```text
                         SATQUERY-AI
                              │
                              ▼
                    Remote-Sensing Input
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
           Single Image   Bi-Temporal   Optical + SAR
                │             │             │
                ▼             ▼             ▼
              RSVQA         CDVQA       CROMA + Qwen
                │             │             │
                └─────────────┼─────────────┘
                              │
                              ▼
                    Natural-Language Answer
```

---

# 🔄 Project Workflow

```text
Satellite Data
      │
      ▼
Image Preprocessing
      │
      ▼
Select Analysis Type
      │
 ┌────┼─────────────┐
 │    │             │
 ▼    ▼             ▼
RSVQA CDVQA    Optical + SAR
 │    │             │
 ▼    ▼             ▼
VQA  Change     Multimodal
     Analysis    Analysis
 │    │             │
 └────┼─────────────┘
      │
      ▼
Question Processing
      │
      ▼
AI Inference
      │
      ▼
Generated Answer
```

---

# 🛠️ Technology Stack

## Programming & Development

- Python
- PyTorch
- JavaScript
- React
- Node.js

## AI / Deep Learning

- Vision Transformers
- BERT
- CROMA
- Qwen
- Vision-Language Models
- Multimodal Feature Fusion
- Change Detection

## Remote Sensing

- Sentinel-1 SAR
- Sentinel-2 Optical Imagery
- GeoTIFF
- Remote-Sensing VQA
- Bi-Temporal Change Detection

## Tools & Platforms

- Git
- GitHub
- Hugging Face
- Google Colab
- Kaggle
- VS Code


---


# 📂 Repository Structure

```text
SatQuery-AI/
│
├── README.md
├── .gitignore
│
├── RSVQA/
│   ├── answer_vocab.json
│   ├── data_pipeline.py
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   ├── README.md
│   └── requirements.txt
│
├── CDVQA/
│   ├── model.py
│   ├── image_loading.py
│   ├── preprocessing.py
│   ├── inference.py
│   ├── inference_utils.py
│   ├── tokenizer.py
│   └── vocabulary.py
│
└── Optical-SAR-VQA/
    ├── croma_config.json
    ├── inference.py
    ├── pretrain_croma.py
    ├── projector_final.pt
    ├── README.md
    │
    └── qwen/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── chat_template.jinja
        ├── README.md
        ├── tokenizer.json
        └── tokenizer_config.json
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/KalaSanthosh104/SatQuery-AI.git
cd SatQuery-AI
```

---

## RSVQA

Navigate to:

```bash
cd RSVQA
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then refer to:

[RSVQA/README.md](RSVQA/README.md)

for model preparation, training, evaluation, and inference instructions.

---

## CDVQA

Navigate to:

```text
CDVQA/
```

The directory contains the implementation for:

- Model architecture
- Image loading
- Preprocessing
- Tokenization
- Vocabulary
- Inference

---

## Optical + SAR VQA

Navigate to:

```text
Optical-SAR-VQA/
```

Refer to:

[Optical-SAR-VQA/README.md](Optical-SAR-VQA/README.md)

for the complete setup and execution procedure.

---

# 💾 Model Checkpoints

Large model checkpoints are kept separate from the main source-code workflow where appropriate.

The CDVQA checkpoint is currently maintained separately as:

```text
best_cdvqa_model.pth.zip
```

This file is intentionally excluded from the Git repository.

---

# 🔮 Future Scope

Future development can include:

- Larger and more diverse remote-sensing datasets
- Improved change detection
- Better multimodal fusion
- Additional satellite modalities
- Improved object-counting performance
- Larger Vision-Language Models
- Faster inference
- Interactive satellite-image querying
- Integration of additional geospatial information
- More robust cross-region generalization

---

# 🎯 Smart India Hackathon 2026

**Problem Statement:** SIH26167

**Project:** SatQuery-AI

SatQuery-AI is developed as a solution for intelligent understanding and querying of remote-sensing imagery using Vision-Language technologies.

---

# 👥 Team

Developed as part of:

**Smart India Hackathon 2026**

---

# 📄 License

This repository is intended for research and educational purposes.

Please refer to the individual model and dataset licenses before using their associated components or data.

---

## ⭐ Acknowledgements

This project builds upon research and open-source technologies in:

- Remote Sensing
- Computer Vision
- Vision-Language Models
- Multimodal Learning
- Satellite Image Analysis
- Change Detection