from flask import Flask, render_template, Response, redirect, url_for
import cv2
from model.test_model import process_frame  

app = Flask(__name__)


cap = None  

def generate_frames():
    """ Capture webcam frames and process them in real-time """
    global cap
    cap = cv2.VideoCapture(0)  

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame)  

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """ Render the home page """
    return render_template('index.html')

@app.route('/detect')
def detect():
    """ Render the detection page """
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    """ Stream video frames to the webpage """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
def stop_detection():
    """ Stop the webcam and return to home page """
    global cap
    if cap is not None:
        cap.release()  
    return redirect(url_for('index'))  

if __name__ == '__main__':
    app.run(debug=True)



