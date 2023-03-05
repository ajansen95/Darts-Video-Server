import random
import time

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'  # for SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

send_scores = False


@socketio.on('connect')
def test_connect():
    print('Client connected')


@socketio.on('disconnect')
def test_disconnect():
    global send_scores
    send_scores = False
    print('Client disconnected')


@socketio.on('ping')
def on_ping():
    print('ping arrived...')
    emit('pong')
    print('sent pong')


@socketio.on('score_request')
def send_score():
    print('sending scores...')
    global send_scores
    send_scores = True

    while send_scores:
        score = {
            "number": random.randint(0, 20),
            "multiplier": random.randint(1, 3),
        }
        emit('score', score)
        print('sent score')
        time.sleep(3)


@socketio.on('score_stop')
def stop_score():
    global send_scores
    send_scores = False
    print('stopped sending scores')


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route("/video")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run()


def generate():
    ret, output_frame = cap.read()
    dimensions = output_frame.shape
    height = dimensions[0]
    width = dimensions[1]
    scale_percent = 100
    # print(dimensions)
    # print("height: " + str(height))
    # print("width: " + str(width))
    # new_dimensions = (int(width * scale_percent / 100), int(height * scale_percent / 100))
    while True:
        # time.sleep(0.1)
        ret, output_frame = cap.read()

        if output_frame is None:
            continue

        # output_frame = cv2.resize(output_frame, new_dimensions)
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        if not flag:
            continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
