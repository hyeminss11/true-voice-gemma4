# true-voice-gemma4
**TrueVoice: Detecting AI Voice-Cloning Scams with Gemma 4 E4B**

Winner, Safety & Trust Track, Gemma 4 Good Hackathon (Google DeepMind × Kaggle)

## Overview
TrueVoice detects AI-generated (deepfake) voices using [Gemma 4 E4B](https://huggingface.co/google/gemma-4-e4b-it)'s audio encoder as a feature extractor. A user receiving a suspicious call can record audio via microphone or upload a file, and receive a verdict (**Real Voice** or **Deepfake Detected**) within seconds. No installation or account is required.

Voice cloning is a useful technology, but its misuse is accelerating: a convincing clone now needs about three seconds of public audio and a free online tool.
- Over **$5 million** lost to AI voice phishing in the US in 2025; 1 in 3 victims lose money, averaging **$18,000 per incident** ([Trend Micro, 2026](https://news.trendmicro.com/2026/04/16/ai-voice-cloning/))
- **$893 million** in AI-related scam losses reported to the FBI in 2025, likely a significant undercount ([FBI IC3, 2025](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf))
- Generative AI-enabled fraud projected to reach **$40 billion by 2027** ([Deloitte, 2024](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html))

Existing detection tools are either enterprise-grade (expensive, API-dependent) or research-grade (not consumer-deployable). TrueVoice is built to fill that gap for the people most targeted, such as elderly users who are least likely to install specialised software.

## Objectives
1. **Transferability**: Can a general-purpose multimodal audio encoder, never trained for anti-spoofing, separate real from synthetic speech?
2. **Real-world robustness**: Does the detector hold up on phone-quality audio rather than studio recordings?
3. **Accessibility**: Can it be delivered in a form that non-technical, high-risk users can actually use?

## Key Findings
- **A frozen encoder is enough.** Training only a **1.5MB classification head** on top of Gemma 4's frozen audio tower reached **5.20% EER** on the full ASVspoof 2021 LA eval set (148,176 samples), with no backbone fine-tuning.
- **Training data must look like the real world.** Codec simulation (16kHz to 8kHz to 16kHz) was the single most impactful preprocessing decision: without it, a model trained on studio audio learns to judge audio quality instead of synthesis artefacts.
- **Linear probing over LoRA.** LoRA fine-tuning runs produced NaN losses; a frozen backbone with a lightweight head trained stably and stays fast enough for browser inference.
- **No install, no account.** A browser URL is a deliberate accessibility decision: the tool needs nothing from the user except a browser and a microphone.

## Dataset
[ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336) Logical Access (train/dev) and ASVspoof 2021 LA (eval).

| Split | Samples | Real | Fake |
|-------|---------|------|------|
| Train | 25,380 | 2,580 | 22,800 |
| Validation | 24,844 | 2,548 | 22,296 |
| Test (ASVspoof 2021 LA eval) | 148,176 | 14,816 | 133,360 |

In addition, deepfake samples of a team member's own voice were generated as a small real-world test set.

> **Citation**:
> J. Yamagishi et al., "ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection", ASVspoof 2021 Workshop, 2021. [arXiv:2109.00537](https://arxiv.org/abs/2109.00537)

## Method

### Architecture
Gemma 4 E4B's `audio_tower` is used as a frozen feature extractor, with a lightweight classification head trained on top.

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
- **Codec simulation**: ASVspoof 2019 recordings are studio-quality, while real phone calls pass through codecs that compress audio to 8kHz. All training audio was downsampled to 8kHz and back to 16kHz, forcing the model to learn genuine voice artefacts rather than audio-quality differences.
- **Linear probing**: Early LoRA fine-tuning runs produced NaN losses. The backbone was therefore frozen entirely and only the classification head was trained.
- **Dtype consistency**: Gemma 4 E4B operates in bfloat16. Input features are explicitly cast to bfloat16 inside the DataCollator to prevent NaN propagation at the boundary with the audio tower.
- **On-the-fly DataCollator**: Loading 25,000+ audio files into RAM is not feasible on Colab, so the dataset stores only file paths and labels and loads audio per batch.

### Training Configuration
| Setting | Value |
|---------|-------|
| Trainable parameters | Classification head only (1.5MB) |
| Epochs | 3 |
| Learning rate | 1e-3 (cosine schedule, 50 warmup steps) |
| Effective batch size | 16 (2 × 8 gradient accumulation) |
| Weight decay | 0.01 |
| Checkpoint selection | Macro F1 on validation |
| Validation macro F1 by epoch | 0.930 → 0.967 → 0.973 |
| Hardware | Google Colab A100 |

## Results Summary
Evaluated on the full [ASVspoof 2021 LA eval set](https://arxiv.org/abs/2109.00537) (148,176 samples):

| Metric | Value |
|--------|-------|
| **EER** | **5.20%** |
| Accuracy | 96% |
| Fake Recall | 96% |
| Real Recall | 93% |
| Fake Precision | 0.99 |
| Classifier size | 1.5MB |

## Team & My Role
Built by a 2-person team. My contributions:
- Generated deepfake samples of my own voice as a real-world test set
- Ran training experiments and hyperparameter tuning, including the switch from LoRA to linear probing after LoRA produced NaN losses
- Ran evaluation on ASVspoof 2021 LA (EER, accuracy, recall, precision)

## Repository Structure
```text
├── notebooks/
│   ├── truevoice_data_preprocess_pipeline.ipynb   # Data preprocessing (incl. codec simulation)
│   ├── truevoice_finetune.ipynb                   # Training (linear probing)
│   ├── truevoice_test.ipynb                       # EER evaluation on ASVspoof 2021 LA
│   └── truevoice_demo.ipynb                       # Gradio demo (run to get a live URL)
└── saved_model/
    ├── classifier_head.pt                         # Trained classification head (1.5MB)
    └── metadata.json                              # Training metadata
```

## Running the Demo
The demo runs as a Gradio web app. Running the final cell in `notebooks/truevoice_demo.ipynb` generates a public URL valid for 72 hours.

**Requirements**
- Google Colab with GPU enabled (T4 or higher)
- Hugging Face account with access to `google/gemma-4-e4b-it`
- HF token set as a Colab secret (`HF_TOKEN`)

**Steps**
1. Download `notebooks/truevoice_demo.ipynb`
2. Upload it to [Google Colab](https://colab.research.google.com)
3. Run all cells in order
4. Copy the public Gradio URL from the output of the last cell

**Supported inputs**
- Upload an audio file (WAV, MP3, FLAC; max 29 seconds)
- Record directly via microphone
- **Phone Call Simulation Mode**: applies 8kHz codec processing to replicate real phone-call conditions

## Loading the Trained Model
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

## Roadmap
- Integration with telecom provider APIs for passive call screening
- On-device inference using Gemma 4 E2B for fully private, offline detection
- Expansion to non-English voice-cloning attacks

## References
- Yamagishi et al., [ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection](https://arxiv.org/abs/2109.00537) (2021)
- FBI, [2025 Internet Crime Report](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf)
- Trend Micro, [AI Voice Cloning Scam Report](https://news.trendmicro.com/2026/04/16/ai-voice-cloning/) (April 2026)
- Deloitte, [Deepfake Banking Fraud Risk on the Rise](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html) (2024)

## License
The project source code is licensed under the MIT License.

For the purposes of the Gemma 4 Good Hackathon, the winning submission and the source code used to generate it are additionally made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) License in accordance with the competition rules.

Third-party models, datasets and libraries remain subject to their respective licenses.
