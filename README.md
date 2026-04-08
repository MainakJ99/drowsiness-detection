# Driver Drowsiness Detection System

## Overview

This project implements a real-time driver drowsiness detection system using a Convolutional Neural Network (CNN) to classify eye states as **alert (open)** or **drowsy (closed)**. The system supports both model training and live webcam-based inference.

The pipeline combines deep learning with classical computer vision (face detection) and incorporates basic temporal smoothing for stable predictions.

---

## Key Features

* CNN-based binary classification (alert vs drowsy)
* Modular pipeline:

  * `train.py` → model training
  * `detect.py` → real-time inference
  * `model.py` → architecture definition
* Real-time webcam detection using OpenCV
* Face-based region extraction for inference
* Training evaluation with accuracy and classification report

---

## Project Structure

```id="s9k2zl"
.
├── train/                     # Dataset (Open_Eyes / Closed_Eyes)
├── model.py                  # CNN architecture
├── train.py                  # Training pipeline
├── detect.py                 # Real-time detection
├── outputs/
│   ├── confusion_matrix.png
│   └── training_curves.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The dataset is organized as:

* `train/Open_Eyes/` → label 0 (alert)
* `train/Closed_Eyes/` → label 1 (drowsy)

Images are converted to grayscale and resized to 64×64 before training.

---

## Installation

Clone the repository:

```id="2rfbde"
git clone https://github.com/your-username/drowsiness-detection.git
cd drowsiness-detection
```

Create a virtual environment:

```id="v3m2vd"
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

Install dependencies:

```id="u4r8p1"
pip install -r requirements.txt
```

---

## Usage

### Train the Model

```id="tq9x0y"
python train.py
```

Outputs:

* Trained model (`.pth`)
* Training curves
* Confusion matrix
* Classification report

---

### Run Real-Time Detection

```id="k0p8ax"
python detect.py
```

Controls:

* Press **Q** to exit webcam

---

## Model Architecture

* Input: grayscale images (64×64)
* 3 convolutional blocks:

  * Conv → BatchNorm → ReLU → Pooling → Dropout
* Adaptive pooling to remove size dependency
* Fully connected classifier

---

## Results

The model is evaluated using:

* Accuracy
* Precision, Recall, F1-score
* Confusion matrix

(Example results should be inserted here for completeness)

---

## System Design Highlights

* Uses face detection to localize region of interest
* Falls back to full-face inference when eyes are not detected
* Applies simple temporal smoothing to reduce prediction noise


---

## Author

Mainak Jana
