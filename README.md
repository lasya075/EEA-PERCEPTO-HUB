# Real-Time Hand Gesture Recognition

This project implements a real-time hand gesture recognition system using **MediaPipe Hands** for precise hand landmark extraction and **machine learning classifiers** for gesture recognition. It can detect 12 common hand gestures based on 3D hand landmark positions.

---

## Project Summary

- **Dataset Size**: Trained on over **12,000 hand sign images**
- **Gestures Supported**:  
  `['Peace', 'Hang Loose', 'Loser', 'High Five', 'You', 'Thumbs Up', 'Thumbs Down', 'Power To', 'OK', 'Good Luck', 'Rock and Roll', 'Call Me']`
-  **Best Accuracy**: **100% on training set**, high test accuracy with RandomForest
-  **Evaluation Metrics**:
  - Accuracy
  - Error Rate
  - mAP (mean Average Precision)
  - Confusion Matrix
-  **Models Used**:  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Random Forest Classifier (best-performing)  

---

## Project Structure
├── data.pkl # Preprocessed dataset (12000 samples)
├── best_model.pkl # Trained ML model saved using pickle
├── mytrain_model.py # Script to train and evaluate models
├── mytest_model.py  # Real-time webcam-based gesture detection
├── mydata_collect.py # (Optional) Data collection script for new signs
├── mycreate_landmarks.py # pickle file of dataset with landmarks
├── app.py                      # Flask GUI application (already built)
├── templates/
    └── detect.html              # Web UI for predicting
    └── index.html
    └── style.css
├── README.md # Project overview

---

## How It Works

1. **MediaPipe Hands** tracks 21 landmarks per hand in real time.
2. Landmark coordinates (x, y, z) are normalized and flattened into a 63-length feature vector.
3. A machine learning model classifies the gesture based on this vector.
4. Predictions are displayed live with bounding boxes on screen.

---

###  Install Dependencies

```bash
pip install opencv-python mediapipe scikit-learn numpy

