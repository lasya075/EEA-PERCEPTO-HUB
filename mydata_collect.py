import os
import cv2

data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
number_of_classes = 5
class_images = 1000

cap = cv2.VideoCapture(0)

for class_index in range(number_of_classes):
    class_dir = os.path.join(data_dir, str(class_index))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Collecting data for class {class_index}.")

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access webcam.")
            break

        cv2.putText(frame, "Press 'm' to start capturing images", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(25) & 0xFF == ord('m'):  # Press 'm' to start capturing images
            done = True

    counter = 0
    while counter < class_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access webcam.")
            break

        cv2.imshow('Webcam', frame)  # Show the frame in the window
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)  # Save the captured frame as an image
        counter += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit capturing images
            break
        
    print(f"Finished capturing images for class {class_index}.")

cap.release()
cv2.destroyAllWindows()
print("Data collection completed.")
