# 🛡️ TrueVoice — Real-Time Deepfake Voice Detection

> *"The technology to clone a voice is accessible to anyone. The technology to detect it has not been — until now."*

[![Model](https://img.shields.io/badge/Model-Gemma%204%20E4B-blue)](https://huggingface.co/google/gemma-4-e4b-it)
[![Dataset](https://img.shields.io/badge/Dataset-ASVspoof%202019%2F2021-green)](https://datashare.ed.ac.uk/handle/10283/3336)
[![EER](https://img.shields.io/badge/EER-5.20%25-orange)](/)
[![Demo](https://img.shields.io/badge/Demo-Gradio-yellow)](/)

---

## The Problem

In July 2025, Sharon Brightwell of Dover, Florida received a call from her "daughter" — crying, claiming she had been in a car accident and needed $15,000 immediately. The voice was indistinguishable from her real daughter's. She sent the money. Her daughter was safe at work the entire time. The voice had been cloned from audio scraped from social media.

This is not an edge case:

- **$16.6 billion** in US cybercrime losses in 2024 (FBI)
- **$5 million+** lost to AI voice cloning scams in 2025 alone
- **1 in 3** people who engage with these calls lose money — averaging **$18,000 per victim**
- **$25.6 million** lost by Arup in a single deepfake video call (February 2024)
- Generative AI-enabled fraud projected to reach **$40 billion globally by 2027** (Deloitte)
- Voice cloning now requires as little as **3 seconds of audio** and a free online tool

---

## What TrueVoice Does

TrueVoice is a real-time deepfake voice detection system. A user receiving a suspicious call opens TrueVoice in their browser, holds the microphone near the speaker, and within seconds receives a verdict: **Real Voice** or **Deepfake Detected**.

No app to install. No account to create. The barrier to protection is a single URL.

---

## How It Works

We repurposed Gemma 4 E4B's dedicated `audio_tower` — an audio encoder trained on vast speech data — as a feature extractor for anti-spoofing, a task it was never explicitly trained for.

**Architecture:**

```
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
```

We froze the audio tower entirely and trained only the **1.5MB classification head**. This is not a limitation — it demonstrates that Gemma 4's audio representations are rich enough to generalize to security-critical tasks without task-specific pretraining of the backbone.

---

## Results

Evaluated on the full **ASVspoof 2021 LA eval set (148,176 samples)**:

| Metric | Value |
|--------|-------|
| **EER** | **5.20%** |
| Accuracy | 96% |
| Fake F1 | 0.98 |
| Real F1 | 0.81 |
| Classifier size | 1.5MB |
| Inference latency | ~2s per clip (A100) |

EER 5.20% with a frozen backbone and a 1.5MB classification head demonstrates the strength of Gemma 4 E4B's audio representations as a general-purpose feature extractor for security-critical downstream tasks.

---

## Training

**Dataset:** ASVspoof 2019 Logical Access (train/dev) + ASVspoof 2021 LA eval

| Split | Samples | Real | Fake |
|-------|---------|------|------|
| Train | 25,380 | 2,580 | 22,800 |
| Validation | 24,844 | 2,548 | 22,296 |
| Test | 148,176 | 14,816 | 133,360 |

**Key design decisions:**

**Codec simulation** — ASVspoof 2019 recordings are studio-quality. Real-world scam calls pass through phone codecs, compressing audio to 8kHz. We applied codec simulation to all training data (downsample 16kHz → 8kHz → 16kHz), forcing the model to learn actual voice artifacts rather than audio quality differences. This is the single most impactful preprocessing decision in our pipeline.

**On-the-fly DataCollator** — Loading 25,000+ audio files into RAM is not feasible on Colab A100. We implemented a batched pipeline: store file paths and labels only, load and process audio per batch, keeping memory stable throughout training.

---

## Challenges

### LoRA Instability — and Why We Dropped It

Our original plan was two-stage: linear probing first, then LoRA fine-tuning of the audio tower. Stage one completed successfully. Stage two became a sustained battle with NaN losses.

The root cause was dtype mixing. Gemma 4 E4B operates in bfloat16. The PEFT library initializes LoRA weights in float32. When these precisions interact inside the audio tower's forward pass, numerical instability produces NaN within the first training epoch:

```
input dtype: bfloat16 ✅
audio_tower output: NaN ← LoRA weights float32 inside bfloat16 tower
```

We attempted full bfloat16 conversion, re-initialization after conversion, lr reduction to 1e-5, and gradient clipping to 0.1. The NaN recurred because `get_peft_model` reinitializes weights after the conversion step, reverting them to float32 each time.

**Decision:** drop LoRA, commit to linear probing only. This was correct.

### Domain Gap Between 2019 and 2021

Without codec simulation, validation accuracy looked strong but test performance on ASVspoof 2021 degraded due to distribution shift between studio-quality training audio and real-world transmission conditions. Codec simulation brought these distributions into alignment.

---

## Technical Stack

| Component | Detail |
|-----------|--------|
| Model | Gemma 4 E4B (`google/gemma-4-e4b-it`) |
| Training | Google Colab A100, HuggingFace Transformers |
| Dataset | ASVspoof 2019 LA + 2021 LA eval |
| Demo | Gradio (browser-accessible URL) |
| Classifier size | 1.5MB |
| Inference latency | ~2s per clip (A100) |

---

## Demo

TrueVoice runs as a Gradio web application accessible from any browser.

**Phone Call Simulation Mode** — applies 8kHz codec processing to replicate real phone call conditions for uploaded files.

**Supported inputs:**
- Upload audio file (WAV, MP3, FLAC — max 29 seconds)
- Record directly via microphone

---

## Repository Structure

```
├── scamshield_data_pipeline.ipynb   # Data preprocessing pipeline
├── scamshield_finetune_gemma4.ipynb # Model training
├── scamshield_test.ipynb            # EER evaluation
├── scamshield_demo.ipynb            # Gradio demo
└── saved_model/
    ├── classifier_head.pt           # Trained classification head (1.5MB)
    └── metadata.json                # Training metadata
```

---

## Reproduce

```python
# 1. Load base model
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor  = AutoProcessor.from_pretrained("google/gemma-4-e4b-it")
base_model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-e4b-it",
    dtype=torch.float32,
    device_map="auto",
)

# 2. Load classifier head
clf_model = AudioDeepfakeClassifier(
    audio_tower=base_model.model.audio_tower,
    hidden_size=1536,
)
clf_model.classifier.load_state_dict(
    torch.load("saved_model/classifier_head.pt")
)
```

---

## Why This Matters

Existing deepfake detection tools are either enterprise-grade (expensive, API-dependent, not consumer-facing) or research-grade and not deployable. TrueVoice is neither.

The elderly are disproportionately targeted by voice phishing (FTC, 2024). They are also the least likely to install specialized software. **A browser URL is a deliberate accessibility decision.**

**Roadmap:**
- Integration with telecom provider APIs for passive, real-time call analysis
- On-device inference using Gemma 4 E2B for fully private, offline detection
- Expansion of training data to non-English voice cloning attacks

---

## References

- Yamagishi et al., ASVspoof 2021 (2021)
- FBI Internet Crime Report 2024
- Trend Micro Voice Cloning Report (2025, 2026)
- Deloitte / BI Intelligence Generative AI Fraud Projection
- FOX 13 Tampa Bay, July 2025
