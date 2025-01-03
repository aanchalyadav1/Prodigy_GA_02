# Image-to-Image Translation with Conditional GAN (Pix2Pix)

## Overview
This project implements an **Image-to-Image Translation Model** using a **Conditional Generative Adversarial Network (cGAN)**. The architecture is based on the Pix2Pix model, a widely recognized approach for paired image translation tasks.

### What is Image-to-Image Translation?
Image-to-Image translation is a computer vision task that transforms an input image into a corresponding output image. Common examples include:
- Converting sketches to photorealistic images
- Translating grayscale images to color
- Changing the style of an image while preserving its content

In this project, we use the **CIFAR-10 dataset** to implement the Pix2Pix model with cGANs.

## Problem Statement
The objective of this project is to:
1. Design a **Generator** capable of creating realistic images conditioned on input noise and class labels.
2. Build a **Discriminator** that distinguishes between real and fake images while considering the given labels.
3. Train both models to enable the Generator to produce convincing images, while the Discriminator becomes adept at identifying fakes.

## Dataset
We use the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images across 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is normalized to the range [-1, 1] for compatibility with the cGAN model.

## Solution
The solution involves implementing a cGAN framework:

### Generator
The Generator takes random noise (`latent vector`) and a class label as input, producing an image corresponding to the given label. The architecture includes:
- An Embedding layer for the class label.
- Dense layers to process noise and label embeddings.
- Conv2DTranspose layers to upsample and generate an image from the combined features.

### Discriminator
The Discriminator evaluates whether an image is real or fake, considering the given label. The architecture includes:
- An Embedding layer for the class label.
- Conv2D layers to process the input image and label embeddings.
- A Dense layer to output a single probability score.

### Loss Functions
- **Binary Cross-Entropy Loss** is used for both Generator and Discriminator.
- The Discriminator minimizes the loss for real images and maximizes it for fake images.
- The Generator minimizes the loss for fake images to fool the Discriminator.

### Optimization
The models are optimized using the Adam optimizer with:
- Learning rate: `0.0002`
- Beta1: `0.5`

## Training Process
The training alternates between:
1. Training the **Discriminator** with real and fake images.
2. Training the **Generator** to produce fake images that can fool the Discriminator.

Each epoch logs the loss values for both models and generates sample images for evaluation.

## Results
The Generator progressively improves during training, producing increasingly realistic images aligned with the input labels. Sample images are saved after each epoch for visual inspection.

## How to Run the Code

### Prerequisites
- Python 3.8+
- TensorFlow 2.9+
- NumPy
- Matplotlib

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pix2pix-cgan.git
   cd pix2pix-cgan
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python train.py
   ```

4. View the generated images in the `output` folder:
   ```bash
   ls output/
   ```

## References
- [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---
Feel free to open issues or submit pull requests to improve the implementation!
