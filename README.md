# Land Use Scene Classification with Deep Learning

This repository contains code and experiments for classifying land use scenes from satellite images using deep learning techniques. The project is based on the [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) and an augmented version available on [Kaggle](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification).

## Project Overview

The goal is to classify satellite images into 21 different land use categories using various neural network architectures, including:

- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Autoencoders
- Transfer Learning with VGG16 and EfficientNetB0

## Project Structure

```
landuse-scene-classification/
├── landuse-scene-classification.ipynb
├── README.md
├── [data and model weights files, not included]
```

## Dataset

- **Source:** [Kaggle: LandUse Scene Classification](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification)
- **Classes:** 21 land use categories (e.g., runway, intersection, agricultural, forest, river, etc.)
- **Images:** 256x256 RGB, 500 images per class (augmented)
- **Splits:** Train (7350), Validation (2100), Test (1050)

## How to Run

1. **Download the Dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification) and unzip.
   - Use the `images_train_test_val` folder for pre-split data.

2. **Install Requirements**
   - Recommended: Use [Kaggle Kernels](https://www.kaggle.com/code) or Google Colab for free GPU.
   - Install required Python packages:
     ```bash
     pip install tensorflow keras opencv-python matplotlib pandas
     ```

3. **Open and Run the Notebook**
   - Open `landuse-scene-classification.ipynb` in Jupyter, VS Code, or Kaggle.
   - Follow the notebook sections to:
     - Explore and preprocess the data
     - Train and evaluate different models
     - Visualize results

## Implemented Models

- **ANN Baseline:** Fully connected network for reference performance.
- **Small CNN:** Basic convolutional network for improved feature extraction.
- **Autoencoder:** For unsupervised feature learning and image reconstruction.
- **VGG16 Transfer Learning:** Uses pre-trained VGG16 as a feature extractor, with custom dense layers for classification.
- **Fine-tuned VGG16:** Unfreezes VGG16 layers for further training on the dataset.
- **Custom Deep CNNs:** With Batch Normalization before/after activation for improved accuracy.
- **EfficientNetB0 Transfer Learning:** State-of-the-art model for high accuracy with fewer parameters.

## Results Summary

- **ANN:** Test accuracy ~0.25 (not suitable for images)
- **Small CNN:** Test accuracy ~0.89
- **VGG16 Transfer Learning:** Test accuracy ~0.88
- **Fine-tuned VGG16:** Test accuracy ~0.91
- **Custom Deep CNNs:** Test accuracy up to ~0.91
- **EfficientNetB0 Transfer Learning:** Test accuracy up to ~0.98

See the notebook for detailed training logs, plots, and comments.

## References

- Jordi Torres, *Python Deep Learning. Introducción Práctica con Keras y TensorFlow 2*, Marcombo (2020).
- Jordi Casas Roma & Anna Bosch Rue, *Fundamentos de las redes neuronales convolucionales*, UOC (2023).
- Mingxing Tan, Quoc V. Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, arXiv:1905.11946 (2019).
- [TensorFlow Keras Transfer Learning Guide](https://www.tensorflow.org/guide/keras/transfer_learning)

---

*This project was developed as part of the Deep Learning course at UOC (M2.975 · Deep Learning · PEC1).*
