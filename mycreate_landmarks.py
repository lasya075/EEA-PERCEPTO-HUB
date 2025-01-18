import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

data_dir = './data'
data = []
labels = []

for dir in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir, dir)):
        auxdata = []

        img = cv2.imread(os.path.join(data_dir, dir, img_path))
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(imgrgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    auxdata.extend([lm.x, lm.y, lm.z])

            data.append(auxdata)
            labels.append(int(dir))

print(f"Data length: {len(data)}")
print(f"Labels length: {len(labels)}")

if len(data) > 0 and len(labels) > 0:
    with open('data.pkl', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data pickled successfully.")
else:
    print("Data or labels are empty!")