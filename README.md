# Driver Drowsiness Detection System

## Overview

This project implements a deep learning-based system to detect driver drowsiness using eye-state classification. The model is trained to classify whether a person is **alert (open eyes)** or **drowsy (closed eyes)** and supports real-time detection using a webcam.

The system operates in two modes:

* **Training Mode** – trains the CNN model on labeled eye images
* **Detection Mode** – performs real-time drowsiness detection

---

## Features

* CNN-based binary classification (Alert vs Drowsy)
* Real-time webcam detection using OpenCV
* Face and eye detection with fallback mechanism
* Temporal smoothing for stable predictions
* Training visualization (loss, accuracy)
* Confusion matrix and classification report

---

## Project Structure

```
.
├── train/
│   ├── Open_Eyes/
│   └── Closed_Eyes/
├── drowsiness.py
├── drowsiness_model.pth
├── confusion_matrix.png
├── training_curves.png
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset should be organized as:

* `Open_Eyes/` → label **0 (alert)**
* `Closed_Eyes/` → label **1 (drowsy)**

---

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/drowsiness-detection.git
cd drowsiness-detection
```

2. Create virtual environment:

```
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```
python drowsiness.py --mode train
```

This will:

* Train the CNN model
* Save the model as `drowsiness_model.pth`
* Generate:

  * `training_curves.png`
  * `confusion_matrix.png`

---

### 2. Run Real-Time Detection

```
python drowsiness.py --mode detect
```

Controls:

* Press **Q** to exit webcam

---

## Model Architecture

* Convolutional Neural Network (CNN)
* Input: grayscale images (64×64)
* 3 convolutional blocks with batch normalization and dropout
* Fully connected classifier

---

## Results

Example outputs:

* Confusion Matrix
* Training Accuracy and Loss Curves
* Real-time detection with probability scores

---

## Key Techniques

* Data augmentation (rotation, flip)
* Dropout regularization
* Learning rate scheduling
* Temporal smoothing for stable predictions



---

## Author

Mainak Jana
