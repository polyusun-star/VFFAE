# VFFAE: Vision-Guided Fashion Fine-Grained Attribute Editing via Semantic Segmentation and Disentangled Representation

This repository provides the official implementation of **VFFAE**, a unified framework for fashion image attribute editing. The framework consists of three main components:

- **FFAS**: Fashion Fine-grained Attribute Segmentation  
- **FSSD**: Fashion Subspace Style Decomposition  
- An end-to-end diffusion-based attribute editing framework

The codebase is implemented using **PyTorch** and **PyTorch Lightning**.

---

## 1. Environment Setup

We recommend using **conda** to configure the environment.

```bash
conda env create -f environment.yaml
conda activate vffae
Please ensure that a CUDA-enabled GPU and a compatible PyTorch version are installed.
```

## 2. FFAS: Fashion Fine-grained Attribute Segmentation

FFAS is designed to generate precise attribute-level segmentation masks guided by textual prompts.

2.1 Training FFAS

To train the FFAS model, simply run:

```bash
python FFAS_train.py
```
Training Dataset
FFAS is trained on the Fashionpedia dataset.

Dataset homepage:
https://fashionpedia.github.io/home/

Please download the dataset and organize it as follows:

```bash
fashionpedia/
├── images/
│   ├── train2020/
│   │   └── train/
│   └── val2020/
│       └── test/
├── instances_attributes_train2020.json
└── instances_attributes_val2020.json
```
You may modify the dataset paths in FFAS_train.py if needed.

2.2 Testing FFAS

To generate attribute masks using a trained FFAS model:

bash
```bash
python FFAS_test.py
```
Before running, please specify the input and output directories in FFAS_test.py:

```bash
image_dir = "your input image directory"
mask_dir = "your output mask directory"
```

2.3 Pretrained FFAS Model

We provide a pretrained FFAS model for convenience:

FFAS pretrained weights:
🔗 [[Download link here]](https://drive.google.com/file/d/1m0vSqP3b-i1dqPh_MiPlmUUth6xXcTeY/view?usp=sharing)

Please place the downloaded weights into the checkpoints/ directory before running the inference script, as follows:

```bash
checkpoints/FFAS_best_model.pth
```

## 3. FSSD: Fashion Style–Structure Disentanglement

FSSD aims to disentangle content and style representations from CLIP-based visual features.

3.1 Training FSSD

To train the FSSD model, run:

```bash
python FSSD_train.py
```
Training Dataset
FSSD is trained on FashionVC dataset organized as:

```bash
data/FashionVCdata/
└── top/
    ├── class1/
    ├── class2/
    └── ...
```
You can replace this dataset with any custom fashion image collection.

3.2 Testing FSSD

To test content–style decomposition and feature swapping:

```bash
python FSSD_test.py
```
Please specify the image paths in FSSD_test.py:

```bash
img1 = load_image("path_to_image_1", ...)
img2 = load_image("path_to_image_2", ...)
```

## 4. Training the Full Framework

The complete VFFAE framework integrates FFAS, FSSD, and a diffusion-based generation model for attribute editing.

4.1 Training

To train the full framework, run:

```bash
bash train.sh
```
or
```bash
python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
--base configs/v1.yaml \
--scale_lr False
```
You can adjust training settings in: configs/v1.yaml
The pretrained model for fine-tuning can be download here: https://drive.google.com/file/d/1JJYfouSK7nC5IKQkxNVGfCk2t5EpD2Ub/view?usp=drive_link

4.2 Testing / Inference

To perform attribute editing inference, run:

```bash
bash test.sh
```
or
```bash
python scripts/inference.py \
--target_region "bag" \
--attribute "style" \
--plms --outdir results/new \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/origin.png \
--reference_path examples/reference/reference.png
```

In the paper, we introduce a natural-language template to describe the editing task:

Change/Replace/Edit the <style/structure/none> of the <target region text> to the reference.

Here we clarify that this template serves **only as a conceptual and user-facing interface**, rather than a model-level conditioning design. 

This template is introduced to illustrate the semantics of the editing task and to unify different attribute editing scenarios in a human-readable way.

In the actual implementation, the framework only relies on two minimal inputs:

1. **Target region text (`target_region`)**  
   - This text is encoded by CLIP and is used **only for open-vocabulary semantic segmentation** (FFAS).
   - It is fully free-form (e.g., `"collar"`, `"neck area"`, `"front neckline"`).

2. **Attribute type (`attribute`)**  
   - The input attribute should be "style", "structure", or empty. 
   - This input is **not encoded by CLIP** and does not affect any text embedding.
   - It acts as a **retrieval key** that selects the corresponding attribute disentanglement branch in FSSD. (seen in cfg.py)

Therefore, in practice, users only need to specify:

```bash
--target_region "bag"
--attribute "style"
```

You can use the trained weights for testing, or use the model weights that we plan to release after the paper is published.

Please place the checkpoint at:
```bash
checkpoints/model.ckpt
```
## 5. Acknowledgements

This repository is developed based on the following open-source projects:

Paint by Example
https://github.com/Fantasy-Studio/Paint-by-Example

CLIPSeg
https://github.com/timojl/clipseg

We sincerely thank the authors for making their code publicly available.

