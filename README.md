# true-voice-gemma4
**TrueVoice: Detecting AI Voice-Cloning Scams with Gemma 4 E4B**

## Overview
This repository contains the code for TrueVoice, built by a 2-person team for the Gemma 4 Good Hackathon (Google DeepMind × Kaggle), where it won the Safety & Trust Track. TrueVoice detects AI-generated (deepfake) voices using [Gemma 4 E4B](https://huggingface.co/google/gemma-4-e4b-it)'s audio encoder as a feature extractor. A user receiving a suspicious call can record audio or upload a file in the browser and receive a verdict (**Real Voice** or **Deepfake Detected**) within seconds, with no installation or account.

Voice cloning now needs only a few seconds of public audio and a free online tool, and AI voice phishing losses are rising quickly. Existing detection tools are either enterprise-grade (expensive, API-dependent) or research-grade (not consumer-deployable), so the people most targeted, such as elderly users, have no practical protection.

## Research Questions
1. **Transferability**: Can a general-purpose multimodal audio encoder, never trained for anti-spoofing, separate real from synthetic speech?
2. **Real-World Robustness**: Does the detector hold up on phone-quality audio rather than studio recordings?
3. **Accessibility**: Can it be delivered in a form that non-technical, high-risk users can actually use?

## Key Findings
- **A frozen encoder is enough**: Training only a 1.5MB classification head on Gemma 4's frozen audio tower reached **5.20% EER** on the full ASVspoof 2021 LA evaluation set (148,176 samples)
- **Training data must look like the real world**: Codec simulation (16kHz → 8kHz → 16kHz) was the most impactful preprocessing decision; without it, a model trained on studio audio learns to judge audio quality instead of synthesis artefacts
- **Linear probing over LoRA**: LoRA fine-tuning runs produced NaN losses; a frozen backbone with a lightweight head trained stably and stays fast enough for browser inference (~2 seconds per clip)
- **No install, no account**: Delivering the detector as a browser URL removes the main barrier for the users most exposed to voice phishing

## Dataset
Training and validation use ASVspoof 2019 Logical Access; evaluation uses the full ASVspoof 2021 Logical Access set.

| Split | Samples | Real | Fake |
|-------|---------|------|------|
| Train (2019 LA) | 25,380 | 2,580 | 22,800 |
| Validation (2019 LA) | 24,844 | 2,548 | 22,296 |
| Test (2021 LA eval) | 148,176 | 14,816 | 133,360 |

Deepfake samples of a team member's own voice were also generated as a small real-world test set.

> **Citation**:
> X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech", Computer Speech & Language, vol. 64, 2020.
> J. Yamagishi et al., "ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection", ASVspoof 2021 Workshop, 2021. [arXiv:2109.00537](https://arxiv.org/abs/2109.00537)

## Model
Gemma 4 E4B's `audio_tower` is used as a frozen feature extractor, and a lightweight classification head is trained on top:

```text
Raw audio (16kHz)
        ↓
Mel spectrogram (via Gemma 4 processor)
        ↓
Gemma 4 E4B audio_tower, frozen (1536-dim output)
        ↓
Mean pooling over time axis
        ↓
Linear(1536→256) → GELU → Dropout(0.3) → Linear(256→2)
        ↓
Real / Fake
```

### Design Decisions
1. **Codec Simulation**: ASVspoof 2019 recordings are studio-quality, while real phone calls are compressed to 8kHz. All training audio was downsampled to 8kHz and back to 16kHz so the model learns genuine voice artefacts rather than audio-quality differences
2. **Linear Probing**: Early LoRA fine-tuning runs produced NaN losses, so the backbone was frozen entirely and only the classification head was trained
3. **Dtype Consistency**: Gemma 4 E4B operates in bfloat16, so input features are explicitly cast to bfloat16 inside the DataCollator to prevent NaN propagation
4. **On-the-fly DataCollator**: The dataset stores only file paths and labels and loads audio per batch, keeping memory stable on Colab

## Results Summary

### ASVspoof 2021 LA Evaluation (148,176 samples)
| Metric | Value |
|--------|-------|
| **EER** | **5.20%** |
| Accuracy | 96% |
| Fake Recall | 96% |
| Real Recall | 93% |
| Fake Precision | 0.99 |
| Classifier Size | 1.5MB |

## My Role
- Generated deepfake samples of my own voice as a real-world test set
- Ran training experiments and hyperparameter tuning, including the switch from LoRA to linear probing after LoRA produced NaN losses
- Ran evaluation on ASVspoof 2021 LA (EER, accuracy, recall, precision)

## Reproducibility

### Training Configuration
| Setting | Value |
|---------|-------|
| Trainable Parameters | Classification head only (1.5MB) |
| Epochs | 3 |
| Learning Rate | 1e-3 (cosine schedule, 50 warmup steps) |
| Effective Batch Size | 16 (2 × 8 gradient accumulation) |
| Weight Decay | 0.01 |
| Checkpoint Selection | Macro F1 on validation |
| Validation Macro F1 by Epoch | 0.930 → 0.967 → 0.973 |
| Hardware | Google Colab A100 |

### Repository Structure
```text
├── notebooks/
│   ├── truevoice_data_preprocess_pipeline.ipynb   # Data preprocessing (incl. codec simulation)
│   ├── truevoice_finetune.ipynb                   # Training (linear probing)
│   ├── truevoice_test.ipynb                       # EER evaluation on ASVspoof 2021 LA
│   └── truevoice_demo.ipynb                       # Gradio demo
└── saved_model/
    └── classifier_head.pt                         # Trained classification head (1.5MB)
```

### Loading the Trained Model
`AudioDeepfakeClassifier` is defined in each notebook (see `notebooks/truevoice_finetune.ipynb`).

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

### Running the Demo
Running all cells in `notebooks/truevoice_demo.ipynb` on Google Colab (GPU, T4 or higher) generates a public Gradio URL valid for 72 hours. A Hugging Face account with access to `google/gemma-4-e4b-it` and an `HF_TOKEN` Colab secret are required.

Supported inputs: audio upload (WAV, MP3, FLAC; max 29 seconds), microphone recording, and **Phone Call Simulation Mode**, which applies 8kHz codec processing to replicate real phone-call conditions.

## Roadmap
- Integration with telecom provider APIs for passive call screening
- On-device inference using Gemma 4 E2B for fully private, offline detection
- Expansion to non-English voice-cloning attacks

## License
The project source code is licensed under the MIT License.

For the purposes of the Gemma 4 Good Hackathon, the winning submission and the source code used to generate it are additionally made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) License in accordance with the competition rules.

Third-party models, datasets and libraries remain subject to their respective licenses.
