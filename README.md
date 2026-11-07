# Age and Gender Prediction Using Deep Learning

A multi-task deep learning project that uses Convolutional Neural Networks (CNNs) to predict **age and gender simultaneously from facial images**.

This project explores two different deep learning approaches: building a CNN from scratch and applying transfer learning using MobileNetV2 pretrained on ImageNet.

## Project Overview

The objective is to develop and compare two CNN architectures capable of performing two tasks simultaneously:

- **Age Estimation:** Predicting a person's age as a regression problem.
- **Gender Classification:** Predicting gender as a binary classification problem.

Both models use the UTKFace facial image dataset and are evaluated using Mean Absolute Error (MAE) for age estimation and classification accuracy for gender prediction.

## Models Implemented

### Model A – Custom CNN

A convolutional neural network designed and trained from scratch.

**Architecture:**
- Four convolutional blocks with 32, 64, 128 and 256 filters.
- 3×3 convolution kernels with ReLU activation.
- 2×2 Max Pooling layers.
- Flatten layer followed by two task-specific dense branches.
- Dropout (0.3) for regularisation.
- Separate output heads for age and gender prediction.

**Training configuration:**
- Optimiser: Adam
- Learning rate: 0.001
- Gender loss: Binary Crossentropy
- Age loss: Mean Absolute Error
- Batch size: 32
- Maximum epochs: 50
- Early Stopping and ReduceLROnPlateau.

### Model B – MobileNetV2 Transfer Learning

A transfer learning approach using MobileNetV2 pretrained on ImageNet.

**Architecture:**
- MobileNetV2 backbone with ImageNet weights.
- Global Average Pooling.
- Shared dense layer with 256 units and ReLU activation.
- Dropout (0.5) for regularisation.
- Separate gender and age prediction branches.

**Two-stage training strategy:**

1. **Head Warm-up:** Freeze the pretrained MobileNetV2 backbone and train the newly added prediction layers for 10 epochs.
2. **Fine-tuning:** Unfreeze approximately the last 30 layers of the backbone and continue training with a reduced learning rate of 0.0001.

The multitask loss uses weights of 0.6 for gender classification and 0.4 for age estimation.

## Dataset

The project uses the **UTKFace dataset**, containing labelled facial images.

The coursework experiments use a subset of 5,000 images.

| Property | Description |
|---|---|
| Dataset | UTKFace |
| Images used | 5,000 |
| Image dimensions | 128 × 128 |
| Colour channels | RGB |
| Training split | 80% (4,000 images) |
| Validation split | 20% (1,000 images) |
| Batch size | 32 |

Age and gender labels are extracted from image filenames following the convention:

`age_gender_race_date.jpg`

Gender labels:
- 0 – Male
- 1 – Female

### Data Preprocessing

- Images are normalised to the [0, 1] range.
- Age labels remain in their original scale.
- Training and validation data are separated using an 80:20 split.
- TensorFlow `tf.data` pipelines are used for efficient batching, shuffling and prefetching.

### Data Augmentation

Lightweight augmentation is applied only to training images:

- Random horizontal flipping.
- Small random rotations (0.03 radians).

This helps improve generalisation while preserving important facial features.

## Results and Performance Analysis

The models are evaluated using two metrics:

- **Gender:** Classification accuracy.
- **Age:** Mean Absolute Error (MAE), measured in years.

### Model A – Custom CNN

| Metric | Training | Validation |
|---|---:|---:|
| Gender Accuracy | ~95% | ~89–90% |
| Age MAE | ~4.8 years | ~7.0–7.2 years |

### Model B – MobileNetV2

| Metric | Training | Validation |
|---|---:|---:|
| Gender Accuracy | ~88–90% | ~86–87% |
| Age MAE | ~4.8–5.0 years | ~7.1–7.3 years |

*Note: These are approximate values reported from the training and validation learning curves, not independently verified held-out test results.*

### Key Findings

- The custom CNN achieved slightly higher validation gender accuracy.
- Both models achieved comparable age estimation performance.
- Both architectures demonstrated reasonable generalisation with relatively small training–validation gaps.
- Transfer learning provided a lightweight alternative to designing a CNN entirely from scratch.

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Google Colab
- MobileNetV2
- Convolutional Neural Networks (CNNs)

## Repository Structure

```text
Age-Gender-Prediction/
│
├── age_gender_notebook.ipynb
│
├── models/
│   ├── age_gender_A.keras
│   └── age_gender_B.keras
│
├── Coursework_report.pdf
│
└── README.md
```

The `models/` directory contains the saved models if you choose to upload them. The dataset is not included in this repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Age-Gender-Prediction.git](https://github.com/Radha-jpg/deep-learning-university.git

cd Age-Gender-Prediction
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib opencv-python
```

### 3. Prepare the dataset

Download the UTKFace dataset and prepare the `train_val/` folder containing the labelled facial images.

The notebook expects the dataset to be available at:

```text
/content/drive/MyDrive/train_val
```

If running outside Google Colab, update the dataset path accordingly.

### 4. Run the notebook

Open `age_gender_notebook.ipynb` using Google Colab or Jupyter Notebook.

For Google Colab:
1. Upload the notebook.
2. Mount Google Drive.
3. Place the dataset in the expected directory.
4. Enable GPU acceleration.
5. Execute the notebook cells sequentially.

The trained models are saved as:

```text
age_gender_A.keras
age_gender_B.keras
```

## Saved Models

The trained model files are available here:

- [Model A – Custom CNN](https://drive.google.com/file/d/1350HvNBrXZHvxLpicCxS3zY6RZJODJxr/view?usp=drive_link)
- [Model B – MobileNetV2](https://drive.google.com/file/d/1tkJLqUISRqQBr5sLbjr9h7Jf15bQ9Lyd/view?usp=drive_link)

## Limitations and Future Improvements

Although both models achieve promising validation results, there are several opportunities for improvement:

- Train on a larger portion of the complete UTKFace dataset.
- Apply stronger but carefully controlled data augmentation.
- Investigate demographic bias across age groups, ethnicities and genders.
- Evaluate performance under different lighting conditions, facial poses and image qualities.
- Explore more advanced architectures and optimisation strategies.
- Conduct independent testing on unseen datasets to assess real-world generalisation.

## Conclusion

This project demonstrates the application of multi-task deep learning for facial age estimation and gender classification.

By implementing both a custom CNN and a MobileNetV2-based transfer learning model, the project compares two approaches to feature extraction, model training and generalisation.

The results highlight how CNN architectures can learn useful facial representations for multiple prediction tasks while demonstrating the benefits and trade-offs of transfer learning.

---

**Author:** Radha Kumari Ray  
**Programme:** MSc Artificial Intelligence  
**Institution:** University of Bath  
**Project:** Understanding Deep Learning – Age and Gender Prediction
