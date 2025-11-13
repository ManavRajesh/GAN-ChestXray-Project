# 🩻 Generating Chest X-Ray Images using GANs

This project explores the use of **Generative Adversarial Networks (GANs)** to create realistic **chest X-ray images** of both healthy subjects and patients with pneumonia.  
The goal is to use synthetic medical images for data augmentation and privacy-preserving AI model development.

---

## 🧠 Project Overview

The proposed model aims to overcome common GAN challenges such as:
- Instability during generator–discriminator training
- Mode collapse (limited diversity of outputs)
- Maintaining perceptual quality and fine-grained structural features

By enforcing stable gradient propagation and optimizing the discriminator to identify fine diagnostic features,  
the generator learns to focus on medically relevant structures during synthesis.

A **conditional GAN (cGAN)** architecture is used, where the discriminator not only determines if an image is real or fake  
but also identifies the underlying medical condition (normal or pneumonia).

> Due to computational limits, the model was trained on **64×64×3 images**.

---

## 📚 Dataset

**Dataset used:** [Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Example images:

![Dataset Sample](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/1218e620-6328-48b1-95d8-20bbbeff794e)

---

## 🏗️ Model Architecture

| Generator Network | Discriminator Network |
|:-----------------:|:---------------------:|
| ![Generator](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/b67effdb-ba3b-470d-9074-91403a25da31) | ![Discriminator](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/eb1fda76-9a03-4ff2-a975-1c4354c9f8d3) |

---

## 🩺 Results

The model successfully generated realistic samples representing both healthy and pneumonia-affected lungs.

### Generated Image Samples
![Generated1](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/e6c8bc72-83c0-4ab3-aa22-3fc29e646efb)
![Generated2](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/f49438f9-4811-4cd2-a869-313defb18fad)

---

## 📈 Evaluation Metrics

To assess the realism of generated images, a secondary **classification network** (based on VGG16) was trained using the synthetic data and tested on the real dataset.

| Classifier | Result |
|:-----------:|:-------:|
| ![Classifier](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/6e556685-5760-4bc0-b3ec-515484313a7e) | ![Performance](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/8c6a7e2f-155b-4e5b-ae26-cb7277488754) |

**Model Performance on Real Dataset (after training on generated images):**
- Accuracy: **93.90%**
- F1-Score: **95.76**
- Recall: **99.12**
- Precision: **92.62**

**Classification Report:**

![Report](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/bc9904b8-6cdb-4c4f-8b8c-f3cd158486b9)

**Confusion Matrix:**

![Confusion](https://github.com/kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images/assets/108609519/cab6ec07-b84b-43cf-b74a-1d9abcbbda53)

---

## 🧾 Summary

This project demonstrates that a well-trained GAN can effectively generate medically relevant chest X-rays.  
Although the classifier showed slight variations in per-class accuracy, this is mainly due to the dataset imbalance between healthy and pneumonia samples.

More extensive training and computational power could further improve output realism and classification consistency.

---

## 🚀 Future Improvements
- Training on higher-resolution images (128×128 or 256×256)
- Using **Conditional GANs** for class-specific generation
- Integrating **Diffusion models** for improved image fidelity
- Performing radiologist-led evaluations for real-world clinical validation

---

## 🧩 Tech Stack
- **Language:** Python 3.x  
- **Framework:** PyTorch  
- **Environment:** Jupyter Notebook / Google Colab  
- **Dataset:** Kaggle Chest X-ray Pneumonia  
- **Evaluation Metrics:** Accuracy, F1, SSIM, FID

---

## 👨‍💻 Authors
**Manav Rajesh (2023BMS-031)**  
**Azzan Syed (2023BMS-031)**  
*Atal Bihari Vajpayee Indian Institute of Information Technology and Management, Gwalior*

---

## 📝 License
This repository is released for academic and educational use.  
If used in research or publications, please provide appropriate citation.

---

### 💬 Acknowledgement
We thank the open-source community and researchers who contributed to the **Chest X-ray Pneumonia Dataset** and foundational **GAN architectures** such as DCGAN and cGAN.

