# 🧠 Brain Tumor Detection Using Deep Learning (EfficientNetB0)

A Streamlit-based web application for automated brain tumor classification from MRI scans using a fine-tuned EfficientNetB0 Convolutional Neural Network.

The application predicts one of four classes:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The model is designed as a Computer-Aided Diagnosis (CAD) tool to assist medical professionals in rapid and reliable MRI image classification.

> **Disclaimer:** This project is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.

---

## Features

- Streamlit-based user interface
- Upload MRI images (PNG/JPG/JPEG)
- Automatic brain tumor classification
- Four-class prediction
- Confidence score visualization
- Deep learning model based on EfficientNetB0
- Fast inference using TensorFlow/Keras

---

## Classes

| Class | Description |
|--------|-------------|
| Glioma | Brain tumor originating from glial cells |
| Meningioma | Tumor arising from the meninges |
| Pituitary | Tumor affecting the pituitary gland |
| No Tumor | Healthy MRI scan |

---

## Model Architecture

The project uses **Transfer Learning** with **EfficientNetB0**.

Architecture:

```
Input Image (512×512)

↓

EfficientNetB0 (Pretrained)

↓

Global Average Pooling

↓

Dropout (0.4)

↓

Dense Softmax Layer (4 Classes)
```

The model was fine-tuned for MRI classification using TensorFlow/Keras. :contentReference[oaicite:1]{index=1}

---

## Dataset

Dataset used:

**Epic + CSCR Hospital Brain MRI Dataset**

Dataset characteristics:

- 12,064 MRI images
- T1-weighted contrast-enhanced MRI
- Image size: 512×512
- Four categories:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor

Dataset split:

- Training: 80%
- Testing: 20%

Please obtain the dataset from its original source if you wish to retrain the model. :contentReference[oaicite:2]{index=2}

---

## Performance

| Metric | Score |
|---------|------:|
| Training Accuracy | 99.18% |
| Validation Accuracy | 98.08% |
| Test Accuracy | 96.00% |
| Macro F1 Score | 0.96 |
| Weighted F1 Score | 0.96 |

These results demonstrate strong classification performance while maintaining balanced precision and recall across all classes. :contentReference[oaicite:3]{index=3}

---

## Project Structure

```
Brain-Tumor-Classification/
│
├── app.py
├── requirements.txt
├── tumor_cnn_f1_score_hospital_model_512.keras
├── .gitignore
├── README.md
│
├── sample_images/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── screenshots/
    ├── home.png
    ├── image-upload.png
    └── prediction-result.png

```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mahak-maheshwari-26/Brain-Tumor-Classification.git
```

Move into the project folder:

```bash
cd <Brain-Tumor-Classification>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Usage

1. Launch the Streamlit application.
2. Upload an MRI image.
3. Click **Analyze Scan**.
4. View the predicted tumor class and confidence score.

---

## Sample Images

A small set of MRI images is included in the `sample_images` directory for quick testing.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- Streamlit
- NumPy
- OpenCV
- Pillow
- Matplotlib

---

## Future Improvements

- Grad-CAM visualization
- Multi-image batch prediction
- PDF report generation
- DICOM image support
- Model explainability
- Cloud deployment

---

## Authors

- **Mahak Maheshwari**

---

## Dataset Link

- **https://data.mendeley.com/datasets/zwr4ntf94j/1**
