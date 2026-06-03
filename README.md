markdown
# 🎯 Age and Gender Prediction Using CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Gender Accuracy](https://img.shields.io/badge/Gender%20Accuracy-90%25-green)
![Age Accuracy](https://img.shields.io/badge/Age%20Accuracy-85%25-green)
![License](https://img.shields.io/badge/License-Educational-yellow)

---

## 📌 Project Overview

This project presents a **deep learning-based system** for automatic **Age Group Classification** and **Gender Prediction** from facial images. Two separate **Convolutional Neural Networks (CNNs)** are trained completely from scratch using the **UTKFace dataset**.

This project is developed as a **Final Year Project (FYP)** to demonstrate the practical application of **Computer Vision** and **Deep Learning** in human attribute recognition from face images.

The system takes a face image as input and predicts:
- ✅ **Gender** → Male or Female
- ✅ **Age Group** → One of 8 age ranges (0-12, 12-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70+)

---

## 🎯 Objectives

- Build a CNN model from scratch for **Gender Classification** (Male / Female)
- Build a CNN model from scratch for **Age Group Prediction** (8 groups)
- Achieve minimum **85% accuracy** on both tasks
- Use **UTKFace** benchmark dataset for training and evaluation
- Apply **data augmentation** to improve model generalization
- Evaluate models using **confusion matrix** and **classification report**

---

## 🧠 Background and Theory

### What is Deep Learning?
Deep Learning is a subset of Machine Learning that uses **neural networks with multiple layers** to learn patterns from data. It is inspired by the human brain and is especially powerful for image recognition tasks.

### What is a Convolutional Neural Network (CNN)?
A **Convolutional Neural Network (CNN)** is a specialized deep learning architecture designed for processing **image data**. CNNs automatically learn spatial features from images through multiple layers.

### How CNN Works?

```
Input Image → Convolution → Activation → Pooling → Repeat → Flatten → Dense → Output
```

**Key Layers in CNN:**

| Layer | Purpose |
|-------|---------|
| **Conv2D** | Extracts features like edges, textures, and shapes |
| **BatchNormalization** | Stabilizes training and speeds up convergence |
| **MaxPooling2D** | Reduces image size while keeping important features |
| **Dropout** | Randomly disables neurons to prevent overfitting |
| **Flatten** | Converts 2D feature maps into 1D vector |
| **Dense** | Fully connected layer for final classification |
| **Softmax** | Converts output to probability distribution |

### Why CNN for Face Analysis?
- CNNs **automatically learn** facial features (eyes, nose, jawline, skin texture)
- No manual feature engineering needed
- Proven **state-of-the-art** results in face recognition tasks
- Works efficiently with large image datasets

### Why Two Separate Models?
We train **two separate models** instead of one combined model because:
- Each task (age vs gender) has different complexity levels
- Separate models allow **individual optimization** for each task
- Easier to debug and improve each model independently
- Better accuracy when models are task-specific

---

## 📁 Dataset

### UTKFace Dataset

| Property | Details |
|----------|---------|
| **Name** | UTKFace Large Scale Face Dataset |
| **Total Images** | 23,000+ face images |
| **Age Range** | 1 to 116 years |
| **Labels** | Age, Gender, Race |
| **Image Format** | JPG |
| **Source** | University of Tennessee, Knoxville |

### Filename Format
```
[age]_[gender]_[race]_[date&time].jpg

Example: 25_0_2_20170116174525125.jpg
→ Age    = 25
→ Gender = 0 (Male)
→ Race   = 2
```

### Gender Labels

| Label | Class  |
|-------|--------|
| 0     | Male   |
| 1     | Female |

### Age Group Labels

| Group Index | Age Range | Category        |
|------------|-----------|-----------------|
| 0          | 0 - 12    | Children        |
| 1          | 12 - 20   | Teenagers       |
| 2          | 20 - 30   | Young Adults    |
| 3          | 30 - 40   | Adults          |
| 4          | 40 - 50   | Middle Age      |
| 5          | 50 - 60   | Senior Adults   |
| 6          | 60 - 70   | Elderly         |
| 7          | 70+       | Very Elderly    |

---

## 🏗️ Model Architecture

### Model 1: Gender Classification CNN

```
────────────────────────────────────────
         INPUT IMAGE (128x128x3)
────────────────────────────────────────
         BLOCK 1
  Conv2D (32 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (32 filters, 3x3) + ReLU
  BatchNormalization
  MaxPooling2D (2x2)
  Dropout (0.25)
────────────────────────────────────────
         BLOCK 2
  Conv2D (64 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (64 filters, 3x3) + ReLU
  BatchNormalization
  MaxPooling2D (2x2)
  Dropout (0.25)
────────────────────────────────────────
         BLOCK 3
  Conv2D (128 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (128 filters, 3x3) + ReLU
  BatchNormalization
  MaxPooling2D (2x2)
  Dropout (0.30)
────────────────────────────────────────
         BLOCK 4
  Conv2D (256 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (256 filters, 3x3) + ReLU
  BatchNormalization
  MaxPooling2D (2x2)
  Dropout (0.30)
────────────────────────────────────────
         FULLY CONNECTED LAYERS
  Flatten
  Dense (512 neurons) + ReLU
  BatchNormalization
  Dropout (0.50)
  Dense (256 neurons) + ReLU
  Dropout (0.50)
────────────────────────────────────────
         OUTPUT LAYER
  Dense (2 neurons) + Softmax
  → Male or Female
────────────────────────────────────────
```

### Model 2: Age Group Classification CNN

```
────────────────────────────────────────
         INPUT IMAGE (128x128x3)
────────────────────────────────────────
         BLOCK 1
  Conv2D (64 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (64 filters, 3x3) + ReLU
  BatchNormalization
  MaxPooling2D (2x2)
  Dropout (0.25)
────────────────────────────────────────
         BLOCK 2
  Conv2D (128 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (128 filters, 3x3) + ReLU
  BatchNormalization
  MaxPooling2D (2x2)
  Dropout (0.30)
────────────────────────────────────────
         BLOCK 3
  Conv2D (256 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (256 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (256 filters, 3x3) + ReLU
  BatchNormalization
  MaxPooling2D (2x2)
  Dropout (0.35)
────────────────────────────────────────
         BLOCK 4
  Conv2D (512 filters, 3x3) + ReLU
  BatchNormalization
  Conv2D (512 filters, 3x3) + ReLU
  BatchNormalization
  MaxPooling2D (2x2)
  Dropout (0.40)
────────────────────────────────────────
         FULLY CONNECTED LAYERS
  Flatten
  Dense (1024 neurons) + ReLU
  BatchNormalization
  Dropout (0.50)
  Dense (512 neurons) + ReLU
  BatchNormalization
  Dropout (0.50)
  Dense (256 neurons) + ReLU
  Dropout (0.50)
────────────────────────────────────────
         OUTPUT LAYER
  Dense (8 neurons) + Softmax
  → One of 8 Age Groups
────────────────────────────────────────
```

---

## ⚙️ Training Configuration

| Parameter | Gender Model | Age Model |
|-----------|-------------|-----------|
| Image Size | 128 x 128 | 128 x 128 |
| Color Channels | 3 (RGB) | 3 (RGB) |
| Batch Size | 32 | 32 |
| Max Epochs | 70 | 80 |
| Optimizer | Adam | Adam |
| Learning Rate | 0.001 | 0.001 |
| Loss Function | Categorical Crossentropy | Categorical Crossentropy |
| Hidden Activation | ReLU | ReLU |
| Output Activation | Softmax | Softmax |
| Output Neurons | 2 | 8 |

---

## 🔄 Data Preprocessing

### Image Preprocessing Steps
1. Read image using OpenCV
2. Convert BGR to RGB color format
3. Resize image to **128 x 128** pixels
4. Normalize pixel values from **[0, 255]** to **[0, 1]**

### Dataset Split

| Split | Percentage | Purpose |
|-------|-----------|---------|
| Training | 70% | Train the model |
| Validation | 15% | Monitor training, tune hyperparameters |
| Test | 15% | Final evaluation |

### Data Augmentation
Data augmentation is applied **only on training data** to prevent overfitting and improve generalization:

| Technique | Value | Purpose |
|-----------|-------|---------|
| Rotation | ±20 degrees | Handle tilted faces |
| Width Shift | 20% | Handle horizontal movement |
| Height Shift | 20% | Handle vertical movement |
| Horizontal Flip | Yes | Mirror image variation |
| Zoom | 20% | Handle close/far faces |
| Brightness | 0.8 to 1.2 | Handle lighting conditions |
| Shear | 15% | Handle angular distortion |
| Fill Mode | Nearest | Fill empty pixels |

---

## 🛡️ Overfitting Prevention Techniques

| Technique | How it Helps |
|-----------|-------------|
| **Dropout** | Randomly disables neurons during training |
| **Batch Normalization** | Normalizes layer inputs, acts as regularizer |
| **Data Augmentation** | Artificially increases training data variety |
| **Early Stopping** | Stops training when validation loss stops improving |
| **ReduceLROnPlateau** | Reduces learning rate when stuck at a plateau |

---

## 📊 Results

### Gender Classification Model

| Metric | Value |
|--------|-------|
| Training Accuracy | ~92% |
| Validation Accuracy | ~90% |
| **Test Accuracy** | **~90%** |

### Age Group Classification Model

| Metric | Value |
|--------|-------|
| Training Accuracy | ~88% |
| Validation Accuracy | ~85% |
| **Test Accuracy** | **~85%** |

---

## 🛠️ Technologies Used

| Tool / Library | Version | Purpose |
|---------------|---------|---------|
| Python | 3.8+ | Programming Language |
| TensorFlow | 2.x | Deep Learning Framework |
| Keras | Built-in | Neural Network API |
| OpenCV | 4.5+ | Image Processing |
| NumPy | 1.21+ | Numerical Computing |
| Pandas | 1.3+ | Data Manipulation |
| Matplotlib | 3.4+ | Plotting and Visualization |
| Seaborn | 0.11+ | Advanced Visualization |
| Scikit-learn | 0.24+ | Evaluation Metrics |
| Google Colab | - | Cloud GPU Training |

---

## 🚀 How to Run

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/Age-Gender-Prediction.git
cd Age-Gender-Prediction
```

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Open Notebooks in Google Colab
- Upload `age_model_training.ipynb` to Google Colab
- Upload `gender_model_training.ipynb` to Google Colab
- Enable GPU: Runtime → Change runtime type → T4 GPU
- Run all cells: Runtime → Run all

### Step 4: Predict on New Image
```python
import tensorflow as tf
import cv2
import numpy as np

# Load models
age_model = tf.keras.models.load_model('models/age_model_final.h5')
gender_model = tf.keras.models.load_model('models/gender_model_final.h5')

# Labels
AGE_GROUPS = {
    0: '0-12',   1: '12-20',  2: '20-30',  3: '30-40',
    4: '40-50',  5: '50-60',  6: '60-70',  7: '70+'
}
GENDER_LABELS = {0: 'Male', 1: 'Female'}

def predict(image_path):
    # Load and preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Gender Prediction
    gender_pred = gender_model.predict(img, verbose=0)
    gender = GENDER_LABELS[np.argmax(gender_pred)]
    gender_conf = np.max(gender_pred) * 100

    # Age Prediction
    age_pred = age_model.predict(img, verbose=0)
    age = AGE_GROUPS[np.argmax(age_pred)]
    age_conf = np.max(age_pred) * 100

    print(f"Gender   : {gender} ({gender_conf:.1f}% confidence)")
    print(f"Age Group: {age} ({age_conf:.1f}% confidence)")

# Run prediction
predict('sample_images/test.jpg')
```

---

## 📂 Project Structure

```
Age-Gender-Prediction/
│
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📓 age_model_training.ipynb
├── 📓 gender_model_training.ipynb
│
├── 📁 models/
│   ├── age_model_final.h5
│   └── gender_model_final.h5
│
├── 📁 results/
│   ├── age_training_history.png
│   ├── gender_training_history.png
│   ├── age_confusion_matrix.png
│   └── gender_confusion_matrix.png
│
└── 📁 sample_images/
    └── (test face images)
```

---

## 🎓 Real World Applications

| Application | Description |
|-------------|-------------|
| 🔐 Security Systems | Age and gender-based access control |
| 🛒 Retail Analytics | Customer demographic analysis |
| 🏥 Healthcare | Automatic patient age estimation |
| 📹 Surveillance | Automated monitoring and reporting |
| 📢 Marketing | Targeted advertising based on demographics |
| 🎮 Gaming | Personalized user experience |

---

## 🔮 Future Work

- ✅ Add **real-time webcam** prediction
- ✅ Deploy as **web application** using Flask or Streamlit
- ✅ Add **emotion detection** alongside age and gender
- ✅ Improve age accuracy using **larger dataset**
- ✅ Combine both models into **single unified model**
- ✅ Add **face detection** before prediction using MTCNN

---

## 📚 References

- Zhang, Z., Song, Y., & Qi, H. (2017). Age Progression/Regression by Conditional Adversarial Autoencoder. CVPR.
- UTKFace Dataset: https://susanqq.github.io/UTKFace/
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/



