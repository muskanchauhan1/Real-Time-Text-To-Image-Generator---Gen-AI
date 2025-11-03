# 🧠 Real Time Text To Image Generator - Gen AI
## 📘 Problem Statement

Traditional image generation models like **Stable Diffusion** can produce generic visuals but struggle to represent **custom concepts** (e.g., a specific person, object, or art style).
This project implements **Textual Inversion**, a fine-tuning technique that teaches the diffusion model a new visual concept by learning a special token (e.g., `<mytoken>`).

Our goal:

> Train a custom embedding to make the model recognize and generate images of a new subject while preserving overall realism and prompt flexibility.

---

## 🧩 Methodology

1. **Model Base:**
   We use the pre-trained **Stable Diffusion v1.5** model from Hugging Face’s `diffusers` library.

2. **Training Technique:**

   * Uses **Textual Inversion**, a parameter-efficient method.
   * Only learns a single token embedding rather than fine-tuning the entire model.
   * The new token (e.g., `<mytoken>`) is then used in text prompts.

3. **Steps Involved:**

   * Prepare 5–10 reference images of your custom subject.
   * Use Hugging Face’s `diffusers` `TextualInversionTrainer` to learn the embedding.
   * Save the output embedding (`learned_embeds.bin`).
   * Load the embedding back into the pipeline for inference.

---

## 🧠 Dataset

* A **custom dataset** of N images (e.g., photos of a specific person or object).
* Images are resized and preprocessed using `torchvision.transforms`.
* For demo purposes, a small synthetic dataset of 5 images was used.

---

## 🧪 Results

| Prompt                                  | Generated Output                        |
| --------------------------------------- | --------------------------------------- |
| “a portrait of `<mytoken>` in a forest” | ![forest](sample_results/prompt1.png)   |
| “<mytoken>` as a digital painting”      | ![painting](sample_results/prompt2.png) |

* The trained embedding successfully captured the visual identity of the subject.
* Model retains coherence across diverse prompts.

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch==2.4.0 torchvision==0.19.0 diffusers==0.31.0 transformers accelerate safetensors
```

---

## 🚀 Usage

### 1️⃣ Clone and Setup

```bash
git clone https://github.com/<muskanchauhan1>/Real Time Text To Image Generator - Gen AI.git
cd Real Time Text To Image Generator - Gen AI
pip install -r requirements.txt
```

### 2️⃣ Run Training

Open the provided notebook:

```
notebooks/Real Time Text To Image Generator - Gen AI.ipynb
```

Update dataset path and token name, then train.

### 3️⃣ Inference

Load your learned embedding:

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
pipe.load_textual_inversion("textual_inversion_output", token="<mytoken>")
prompt = "a photo of <mytoken> in a forest"
image = pipe(prompt).images[0]
image.save("output.png")
```

---

## 📊 Future Work

* Compare Textual Inversion with **DreamBooth** and **LoRA** fine-tuning.
* Extend dataset for multi-style learning.
* Evaluate embedding stability across diffusion model versions.

---

## 👩‍💻 Author

**MUSKAN CHAUHAN**
Post Graduate Diploma in Big Data Analytics (PG-DBDA)
C-DAC Mumbai

---

## 🪪 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
