from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib


app = Flask(__name__)
  
# Connects to your computer's default camera
cap = cv2.VideoCapture(0)
  
# Detect the coordinates
detector = dlib.get_frontal_face_detector()

@app.route('/')
def index():
    return render_template('index.html')
  
def gen_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Iterator to count faces
        i = 0
        for face in faces:
            # Get the coordinates of faces
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            # Increment iterator for each face in faces
            i = i+1

            # Display the box and faces
            cv2.putText(frame, 'face num'+str(i), (x-10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(face, i)

        # Display the resulting frame
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/FaceCount')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
  
# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
