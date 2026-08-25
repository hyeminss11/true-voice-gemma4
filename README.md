# 🛡️ TrueVoice — Deepfake Voice Detection with Gemma 4 E4B

## Overview

TrueVoice detects AI-generated (deepfake) voices using [Gemma 4 E4B](https://huggingface.co/google/gemma-4-e4b-it)'s audio encoder as a feature extractor. A user receiving a suspicious call can record audio via microphone or upload a file, and receive a verdict — **Real Voice** or **Deepfake Detected** — within seconds. No installation or account required.

---

## The Problem

Voice cloning is a genuinely useful technology, but its misuse is accelerating. A convincing voice clone now requires three seconds of publicly available audio and a free online tool.

- Over **$5 million** lost to AI voice phishing in the US in 2025; 1 in 3 victims lose money, averaging **$18,000 per incident** ([Trend Micro, 2026](https://news.trendmicro.com/2026/04/16/ai-voice-cloning/))
- **$893 million** in AI-related scam losses reported to the FBI in 2025 — likely a significant undercount ([FBI IC3, 2025](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf))
- Generative AI-enabled fraud projected to reach **$40 billion by 2027** ([Deloitte, 2024](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html))

Existing detection tools are either enterprise-grade (expensive, API-dependent) or research-grade (not consumer-deployable). TrueVoice fills that gap.

---

## Architecture

We repurposed Gemma 4 E4B's `audio_tower` as a frozen feature extractor and trained a lightweight 1.5MB classification head on top.

```text
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

We froze the audio tower entirely and trained only the **1.5MB classification head**. This demonstrates that Gemma 4's pretrained audio representations generalize to security-critical tasks without backbone fine-tuning.

---

## Results

Evaluated on the full [ASVspoof 2021 LA eval set](https://arxiv.org/abs/2109.00537) (148,176 samples):

| Metric | Value |
|--------|-------|
| **EER** | **5.20%** |
| Accuracy | 96% |
| Fake Recall | 96% |
| Real Recall | 93% |
| Fake Precision | 0.99 |
| Classifier size | 1.5MB |

---

## Training

**Dataset:** [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336) Logical Access (train/dev) + ASVspoof 2021 LA eval ([Yamagishi et al., 2021](https://arxiv.org/abs/2109.00537))

| Split | Samples | Real | Fake |
|-------|---------|------|------|
| Train | 25,380 | 2,580 | 22,800 |
| Validation | 24,844 | 2,548 | 22,296 |
| Test | 148,176 | 14,816 | 133,360 |

**Key design decisions:**

**Codec simulation** — ASVspoof 2019 recordings are studio-quality, while real-world phone calls pass through codecs that compress audio to 8kHz. We applied codec simulation to all training data (16kHz → 8kHz → 16kHz), forcing the model to learn actual voice artifacts rather than audio quality differences. This is the single most impactful preprocessing decision in our pipeline.

**On-the-fly DataCollator** — Loading 25,000+ audio files into RAM is not feasible on Colab. We store only file paths and labels in the dataset object and load audio per batch, keeping memory stable throughout training.

**Dtype consistency** — Gemma 4 E4B operates in bfloat16. Input features must be explicitly cast to bfloat16 inside the DataCollator to prevent NaN propagation at the boundary with the audio tower's computation graph.

---

## Repository Structure

```text
├── truevoice_data_pipeline.ipynb   # Data preprocessing pipeline
├── truevoice_finetune.ipynb        # Model training (linear probing)
├── truevoice_test.ipynb            # EER evaluation on ASVspoof 2021
├── truevoice_demo.ipynb            # Gradio demo (run to get live URL)
└── saved_model/
    ├── classifier_head.pt          # Trained classification head (1.5MB)
    └── metadata.json               # Training metadata
```

---

## Running the Demo

The demo runs as a Gradio web app. Running the final cell in `truevoice_demo.ipynb` generates a public URL valid for 72 hours.

**Requirements:**
- Google Colab with GPU enabled (T4 or higher)
- HuggingFace account with access to `google/gemma-4-e4b-it`
- HF token set as Colab secret (`HF_TOKEN`)

**Steps:**
1. Download `truevoice_demo.ipynb` from this repository
2. Upload to [Google Colab](https://colab.research.google.com)
3. Run all cells in order
4. Copy the public Gradio URL from the output of the last cell

**Supported inputs:**
- Upload audio file (WAV, MP3, FLAC — max 29 seconds)
- Record directly via microphone
- **Phone Call Simulation Mode**: applies 8kHz codec processing to replicate real phone call conditions

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
| Training | Google Colab, HuggingFace Transformers |
| Dataset | ASVspoof 2019 LA + 2021 LA eval |
| Demo | Gradio (browser-accessible, 72hr public URL) |
| Classifier size | 1.5MB |

---

## Why This Matters

The elderly are disproportionately targeted by voice phishing and are also the least likely to install specialized software. A browser URL is a deliberate accessibility decision — TrueVoice requires nothing from the user except a browser and a microphone.

**Roadmap:**
- Integration with telecom provider APIs for passive call screening
- On-device inference using Gemma 4 E2B for fully private, offline detection
- Expansion to non-English voice cloning attacks

---

## References

- Yamagishi et al., [ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection](https://arxiv.org/abs/2109.00537) (2021)
- FBI, [2025 Internet Crime Report](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf)
- Trend Micro, [AI Voice Cloning Scam Report](https://news.trendmicro.com/2026/04/16/ai-voice-cloning/) (April 2026)
- Deloitte, [Deepfake Banking Fraud Risk on the Rise](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html) (2024)

## License
The project source code is licensed under the MIT License.

For the purposes of The Gemma 4 Good Hackathon, the winning Submission and the source code used to generate it are additionally made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) License in accordance with the competition rules.

Third-party models, datasets, and libraries remain subject to their respective licenses.
