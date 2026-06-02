# 🐺 DreamBooth LoRA — Few-Shot Subject-Driven Image Generation on SDXL

> Fine-tuning Stable Diffusion XL to generate specific real-world subjects (a wolf plushie and a grey sloth plushie) in any scenario — using only 5 real photos each, trained on a single 15GB Kaggle T4 GPU.

![Scientific Comparison Grid](results/scientific_comparison_grid.png)

---

## 📌 Project Overview

This project implements a complete **DreamBooth LoRA fine-tuning pipeline** on **Stable Diffusion XL (SDXL 1.0)** for subject-driven text-to-image generation.

The core challenge: teach a 6.6 billion parameter generative model to recognize and reproduce two specific physical toys it has never seen — using only 5 training photos per subject — on hardware that cannot even run SDXL in its standard configuration.

**Subjects trained:**
- `sks` → Wolf Plushie Toy
- `xyz` → Grey Sloth Plushie Toy

**Key Result:** CLIP-I fidelity score improved from **48.20% (SD 1.5 baseline)** to **70.88% (our SDXL LoRA)** — a 47% relative improvement.

---

## 🗂️ Repository Structure

```
dreambooth-lora-sdxl/
│
├── dreambooth_lora_sdxl.ipynb       ← Main notebook (all 18 cells)
│
├── results/
│   ├── scientific_comparison_grid.png     ← Base vs Wolf vs Sloth LoRA (3×3)
│   ├── wolf_lora_results.png              ← Wolf LoRA across 6 prompts
│   ├── sloth_lora_results.png             ← Sloth LoRA across 6 prompts
│   ├── generational_evolution.png         ← SD 1.5 → SDXL → LoRA evolution
│   ├── base_vs_finetuned_ablation.png     ← Before vs after fine-tuning
│   ├── merged_lora_crossover.png          ← Wolf + Sloth dual LoRA experiment
│   ├── ultimate_crossover_success.png     ← Best merged LoRA result
│   └── failure_cases_analysis.png         ← Honest limitations analysis
│
└── README.md
```

---

## ⚙️ Pipeline Architecture

The full pipeline runs across 18 notebook cells in 5 distinct phases:

### Phase 1 — Environment Setup (Cells 1–3)
- Clones and installs diffusers from source
- Configures Accelerate for single-GPU FP16 training
- Authenticates with Hugging Face Hub
- Downloads the Google DreamBooth dataset (wolf + sloth subjects)

### Phase 2 — Data Augmentation (Cells 4–7)
- Generates **100 class images per subject** using SDXL (prior preservation)
- Generates **20 diverse backgrounds** using SD 1.5
- Extracts subjects from real photos using **rembg (IS-Net segmentation)**
- Composites extracted subjects onto synthetic backgrounds with edge feathering
- Result: **5 real + 20 synthetic = 25 training images per subject**

### Phase 3 — LoRA Training (Cell 8)
- Trains separate Rank-32 LoRA adapters for both subjects
- Full optimization stack for 15GB VRAM constraint (see Memory Optimization section)
- Pushes trained weights to Hugging Face Hub automatically

### Phase 4 — Inference & Evaluation (Cells 9–17)
- Single-subject generation with SDXL Base + Refiner ensemble
- Base vs fine-tuned ablation comparison
- Multi-LoRA joint generation (both subjects simultaneously)
- Generational evolution comparison (SD 1.5 → SDXL → LoRA)
- Failure case stress testing (spatial, scale, complex interaction)

### Phase 5 — Quantitative Evaluation (Cell 18)
- CLIP-I similarity scoring using `openai/clip-vit-base-patch32`
- Executive dashboard with hyperparameter matrix and benchmark table

---

## 🧠 Key Technical Contributions

### 1. Synthetic Data Augmentation
Standard DreamBooth with 5 images causes severe overfitting to training backgrounds. This pipeline:
- Removes backgrounds with IS-Net segmentation (rembg)
- Feathers alpha edges with Gaussian blur (radius=1.5) for photorealistic compositing
- Composites subjects onto 20 diverse AI-generated environments
- Validates mask quality (coverage check: 10%–85%) before augmenting

### 2. Memory Optimization Stack (6.6B params on 15GB)
Five simultaneous techniques to run SDXL on a T4:

| Technique | VRAM Saving | Implementation |
|---|---|---|
| 8-bit Adam (bitsandbytes) | ~75% optimizer memory | `--use_8bit_adam` |
| Gradient Checkpointing | ~40% activation memory | `--gradient_checkpointing` |
| FP16 Mixed Precision | ~50% tensor size | `--mixed_precision=fp16` |
| Fixed FP16 VAE | Prevents FP16 overflow | `madebyollin/sdxl-vae-fp16-fix` |
| CPU Offload at Inference | Peak VRAM ~3GB | `enable_model_cpu_offload()` |

### 3. SDXL Base + Refiner Ensemble
- Base model runs denoising from step 0 → 80% (`denoising_end=0.8`, output_type="latent")
- Refiner takes over from 80% → 100% (`denoising_start=0.8`)
- `text_encoder_2` and VAE shared between both models to save ~8GB VRAM

### 4. Multi-LoRA Joint Generation
Both trained adapters loaded simultaneously with blended weights:
```python
base.load_lora_weights("./lora_wolf_plushie_sdxl_r32", adapter_name="wolf")
base.load_lora_weights("./lora_grey_sloth_plushie_sdxl_r32", adapter_name="sloth")
base.set_adapters(["wolf", "sloth"], adapter_weights=[0.6, 0.6])
```

---

## 📊 Results

### Quantitative Benchmark

| Model | Training Data | VRAM Required | CLIP-I Score | Status |
|---|---|---|---|---|
| Stable Diffusion 1.5 | 5 real images | ~6.0 GB | 48.20% | ❌ Poor fidelity |
| SDXL 1.0 (standard) | 5 real images | ~24.0 GB | — | ❌ OOM crash |
| **SDXL + LoRA (ours)** | **5 real + 20 augmented** | **13.5 GB** | **70.88%** | **✅ Success** |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Base Model | stabilityai/stable-diffusion-xl-base-1.0 |
| LoRA Rank | 32 |
| Learning Rate | 1e-4 |
| LR Scheduler | Cosine |
| Warmup Steps | 100 |
| Max Train Steps | 600 |
| Resolution | 768px |
| Batch Size | 1 (grad accumulation: 4) |
| Prior Loss Weight | 1.0 |
| Seed | 42 |

---

## 🖼️ Visual Results

### Scientific Comparison Grid (Base vs Wolf LoRA vs Sloth LoRA)
Three scenarios tested: On the Moon · As a Samurai · Underwater

### Generational Evolution
SD 1.5 (failed geometry) → SDXL Base (generic plushie) → SDXL + LoRA (exact subject)

### Failure Cases (Honest Limitations)
| Failure Type | Cause |
|---|---|
| Spatial/Pose (upside down) | LoRA has no data of subject from below |
| Scale/Anatomy (extreme macro) | No high-res texture training data |
| Complex Interaction (on bike) | Cannot model rigid plushie limb physics |

---

## 🔧 How to Run

### Requirements
- Kaggle account with GPU T4 x2 enabled
- Hugging Face account + token (stored as Kaggle secret `HF_TOKEN`)
- ~15GB VRAM minimum

### Steps
1. Upload `dreambooth_lora_sdxl.ipynb` to Kaggle
2. Enable GPU accelerator (T4 x2 recommended)
3. Add your HuggingFace token as a Kaggle secret named `HF_TOKEN`
4. Run cells 1–8 sequentially for full training
5. Run cells 10–18 for inference and evaluation
6. Delete cells 9, 13, and 19 (duplicates/empty) before submission

### Environment
```
Platform:     Kaggle (Ubuntu 20.04)
GPU:          NVIDIA T4 (15GB VRAM)
Python:       3.10
PyTorch:      2.x
diffusers:    from source (huggingface/diffusers)
PEFT:         >=0.17.0
```

---

## 📚 References & Credits

- **DreamBooth**: Ruiz et al. (2022) — [arxiv.org/abs/2208.12242](https://arxiv.org/abs/2208.12242)
- **LoRA**: Hu et al. (2021) — [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- **SDXL**: Podell et al. (2023) — [arxiv.org/abs/2307.01952](https://arxiv.org/abs/2307.01952)
- **CLIP**: Radford et al. (2021) — OpenAI
- **IS-Net (rembg)**: [github.com/xuebinqin/DIS](https://github.com/xuebinqin/DIS)
- **DreamBooth Dataset**: [google/dreambooth](https://github.com/google/dreambooth)
- **Diffusers Library**: Hugging Face

---

## 👤 Author

**Taha Tanvir**
---

## 📄 License

This project is for academic and educational purposes.
Model weights follow the [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
