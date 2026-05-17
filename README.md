# 🛡️ TrueVoice — Deepfake Voice Detection with Gemma 4 E4B

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyeminss11/true-voice-gemma4/blob/main/truevoice_demo.ipynb)
[![Model](https://img.shields.io/badge/Model-Gemma%204%20E4B-blue)](https://huggingface.co/google/gemma-4-e4b-it)
[![Dataset](https://img.shields.io/badge/Dataset-ASVspoof%202019%2F2021-green)](https://datashare.ed.ac.uk/handle/10283/3336)
[![EER](https://img.shields.io/badge/EER-5.20%25-orange)](/)

---

## Overview

TrueVoice detects AI-generated (deepfake) voices using Gemma 4 E4B's audio encoder as a feature extractor. A user receiving a suspicious call can record the audio via microphone or upload a file, and receive a verdict — Real Voice or Deepfake Detected — within seconds. No installation or account required.

---

## The Problem

Voice cloning now requires three seconds of publicly available audio and a free online tool. The misuse of this technology is accelerating: AI-powered voice phishing attacks resulted in over $5 million in documented US losses in 2025, with 1 in 3 victims losing money and average losses of $18,000 per victim (Trend Micro, 2026). The FBI's 2025 IC3 report flagged $893 million in AI-related scam losses, and Deloitte projects generative AI-enabled fraud will reach $40 billion by 2027.

Existing detection tools are either enterprise-grade (expensive, API-dependent) or research-grade (not consumer-deployable). TrueVoice fills that gap with a browser-accessible tool requiring nothing from the user except a microphone.

---

## Architecture

We repurposed Gemma 4 E4B's `audio_tower` as a frozen feature extractor and trained a lightweight 1.5MB classification head on top.
Raw audio (16kHz)
↓
Mel spectrogram (via Gemma 4 processor)
↓
Gemma 4 E4B audio_tower — frozen (1536-dim output)
↓
Mean pooling over time axis
↓
Linear(1536→256) → GELU → Dropout(0.3) → Linear(256→2)
↓
Real / Fake

---

## Results

Evaluated on the full ASVspoof 2021 LA eval set (148,176 samples):

| Metric | Value |
|--------|-------|
| **EER** | **5.20%** |
| Accuracy | 96% |
| Fake Recall | 96% |
| Real Recall | 93% |
| Fake Precision | 0.99 |
| Classifier size | 1.5MB |
| Inference latency | ~2s per clip (A100) |

---

## Training

**Dataset:** ASVspoof 2019 Logical Access (train/dev) + ASVspoof 2021 LA eval

| Split | Samples | Real | Fake |
|-------|---------|------|------|
| Train | 25,380 | 2,580 | 22,800 |
| Validation | 24,844 | 2,548 | 22,296 |
| Test | 148,176 | 14,816 | 133,360 |

**Key design decisions:**

**Codec simulation** — ASVspoof 2019 recordings are studio-quality, while real-world phone calls pass through codecs that compress audio to 8kHz. We applied codec simulation to all training data (16kHz → 8kHz → 16kHz), forcing the model to learn actual voice artifacts rather than audio quality differences. This is the single most impactful preprocessing decision in our pipeline.

**On-the-fly DataCollator** — Loading 25,000+ audio files into RAM is not feasible on Colab A100. We store only file paths and labels in the dataset object and load audio per batch, keeping memory stable throughout training.

**Dtype consistency** — Gemma 4 E4B operates in bfloat16. Input features must be explicitly cast to bfloat16 inside the DataCollator to prevent NaN propagation at the boundary with the audio tower's computation graph.

---

## Repository Structure
├── truevoice_data_pipeline.ipynb   # Data preprocessing pipeline
├── truevoice_finetune.ipynb        # Model training (linear probing)
├── truevoice_test.ipynb            # EER evaluation on ASVspoof 2021
├── truevoice_demo.ipynb            # Gradio demo (run to get live URL)
└── saved_model/
├── classifier_head.pt          # Trained classification head (1.5MB)
└── metadata.json               # Training metadata

---

## Running the Demo

The demo runs as a Gradio web app. Running the final cell in `truevoice_demo.ipynb` generates a public URL valid for 72 hours.

**Requirements:**
- Google Colab (A100 recommended) or local GPU with 16GB+ VRAM
- HuggingFace account with access to `google/gemma-4-e4b-it`
- HF token set as Colab secret (`HF_TOKEN`)

**Steps:**
1. Open `truevoice_demo.ipynb` in Colab (click the badge above)
2. Mount Google Drive and set paths
3. Run all cells in order
4. Copy the public Gradio URL from the output of the last cell

**Supported inputs:**
- Upload audio file (WAV, MP3, FLAC — max 29 seconds)
- Record directly via microphone
- Phone Call Simulation Mode: applies 8kHz codec processing to replicate real phone call conditions

---

## Reproducing Training

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor = AutoProcessor.from_pretrained("google/gemma-4-e4b-it")
base_model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-e4b-it",
    dtype=torch.bfloat16,
    device_map="auto",
)

clf_model = AudioDeepfakeClassifier(
    audio_tower=base_model.model.audio_tower,
    hidden_size=1536,
)
clf_model.classifier.load_state_dict(
    torch.load("saved_model/classifier_head.pt")
)
clf_model = clf_model.to(torch.bfloat16)
```

Full training pipeline: see `truevoice_finetune.ipynb`

---

## Technical Stack

| Component | Detail |
|-----------|--------|
| Model | Gemma 4 E4B (`google/gemma-4-e4b-it`) |
| Training | Google Colab A100, HuggingFace Transformers |
| Dataset | ASVspoof 2019 LA + 2021 LA eval |
| Demo | Gradio (browser-accessible, 72hr public URL) |
| Classifier size | 1.5MB |
| Inference latency | ~2s per clip (A100) |

---

## References

- Yamagishi et al., [ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection](https://arxiv.org/abs/2109.00537) (2021)
- FBI, [2025 Internet Crime Report](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf)
- Trend Micro, [AI Voice Cloning Scam Report](https://news.trendmicro.com/2026/04/16/ai-voice-cloning/) (April 2026)
- Deloitte, [Deepfake Banking Fraud Risk on the Rise](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html) (2024)
