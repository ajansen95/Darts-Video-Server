import random
import time

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import math


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

@app.route("/manipulated_video")
def manipulated_video():
    return Response(manipulate(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
        

#############################################################################################
##################### OpenCV Part ###########################################################
#############################################################################################

# Color ranges for the Masks in hsv color system
# green mask
lower_green = np.array([60, 50, 50])
upper_green = np.array([80, 255, 255])

# lower red mask (0-10)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

# upper red mask (165-180)
lower_red2 = np.array([165, 50, 50])
upper_red2 = np.array([180, 255, 255])

# mapped dartboard
centre = (452, 452)
middle_up = (452, 111)
angles = [(351, 9, 20), (9, 27, 5), (27, 45, 12), (45, 63, 9), (63, 81, 14), (81, 99, 11), (99, 117, 8),
          (117, 135, 16),
          (135, 153, 7), (153, 171, 19), (171, 189, 3), (189, 207, 17), (207, 225, 2), (225, 243, 15),
          (243, 261, 10), (261, 279, 6), (279, 297, 13), (297, 315, 4), (315, 333, 18), (333, 351, 1)]

distance = [(0, 15, 50), (16, 31, 25), (32, 194, 1), (194, 215, 3), (216, 325, 1), (326, 340, 2)]

object_detector = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=60, detectShadows=False)

coords = []
point_history = []
dart_thrown = False
counter = 0
current_point = None



# Calibrate a given frame
def calibrate_dartboard(frame):
    board_points = [[453, 120], [785, 450], [453, 785], [120, 450]]
    cam_points = []

    # Put masks on the frame
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    red_mask = cv2.inRange(img_hsv, lower_red1, upper_red1) + cv2.inRange(img_hsv, lower_red2, upper_red2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coords = []

    # Iterate through all contours and append the mid of each contour in an array
    # First for the contours of the red mask
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            coords.append((int(y + (h / 2)), int(x + (w / 2))))

    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coords2 = []

    # Iterate through the green mask
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            coords2.append((int(x + (w / 2)), int(y + (h / 2))))

    # add calibration Points to the array
    # 20, 6, 3, 11 in that order

    cam_points.append((min(coords)[1], min(coords)[0]))
    cam_points.append(max(coords2))
    cam_points.append((max(coords)[1], max(coords)[0]))
    cam_points.append(min(coords2))

    # Calibrate the Image via these two arrays
    board_points = np.array([board_points[0], board_points[1], board_points[2], board_points[3]], dtype=np.float32)
    cam_points = np.array([cam_points[0], cam_points[1], cam_points[2], cam_points[3]], dtype=np.float32)

    cam_to_board = cv2.getPerspectiveTransform(cam_points, board_points)
    return cam_to_board


# Detect dart tip and return the coordinates of it
def dart_detection(contours):
    global dart_thrown
    global counter
    global coords
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            counter = 0
            dart_thrown = True
            for c in cnt:
                x, y = c[0]
                coords.append((x, y))


# Applies a circular Maks on the incoming image
def circular_mask(frame):
    mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
    cv2.circle(mask, (452, 452), 435, (255, 255, 255), -1, 8, 0)
    out = frame * mask
    white = 255 - mask
    stream = white - out
    return stream


# Calculates the Point value with a given Coordinate
def get_point_value(coord):
    cv2.waitKey(0)
    dst = math.dist(centre, coord)

    lineA = (centre, middle_up)
    lineB = (centre, coord)

    line1Y1 = lineA[0][1]
    line1X1 = lineA[0][0]
    line1Y2 = lineA[1][1]
    line1X2 = lineA[1][0]

    line2Y1 = lineB[0][1]
    line2X1 = lineB[0][0]
    line2Y2 = lineB[1][1]
    line2X2 = lineB[1][0]

    # calculate angle between pairs of lines
    angle1 = math.atan2(line1Y1 - line1Y2, line1X1 - line1X2)
    angle2 = math.atan2(line2Y1 - line2Y2, line2X1 - line2X2)
    angleDegrees = int((angle1 - angle2) * 360 / (2 * math.pi))
    if angleDegrees < 0:
        angleDegrees += 360

    points = 0
    multiplier = 0
    for d in distance:
        if dst > 340:
            print(str(multiplier) + " * " + str(points))
            return 0, 0
        if dst <= d[1]:
            multiplier = d[2]
            if d[2] == 25 or d[2] == 50:
                print(str(multiplier) + " * " + str(points))
                return multiplier, 0
            break

    for ang in angles:
        if angleDegrees >= ang[0]:
            if angleDegrees < ang[1]:
                points = ang[2]
                break
    if angleDegrees > 351:
        points = 20
    if angleDegrees < 9:
        points = 20

    print(str(multiplier) + " * " + str(points))
    return multiplier, points


# test functions
def calibration_points(frame):
    # Put masks on the frame
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    red_mask = cv2.inRange(img_hsv, lower_red1, upper_red1) + cv2.inRange(img_hsv, lower_red2, upper_red2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coords = []
    # Iterate through all contours and append the mid of each contour in an array
    # First for the contours of the red mask
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            coords.append((int(y + (h / 2)), int(x + (w / 2))))

    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coords2 = []
    # Iterate through the green mask
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            coords2.append((int(x + (w / 2)), int(y + (h / 2))))

    cv2.drawMarker(frame, (min(coords)[1], min(coords)[0]), (123, 255, 123), cv2.MARKER_CROSS, 20, 5)
    cv2.drawMarker(frame, max(coords2), (123, 255, 123), cv2.MARKER_CROSS, 20, 5)
    cv2.drawMarker(frame, (max(coords)[1], max(coords)[0]), (123, 255, 123), cv2.MARKER_CROSS, 20, 5)
    cv2.drawMarker(frame, min(coords2), (123, 255, 123), cv2.MARKER_CROSS, 20, 5)

    return frame


def get_mask(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    red_mask = cv2.inRange(img_hsv, lower_red1, upper_red1) + cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)
    green_result = cv2.bitwise_and(frame, frame, mask=green_mask)
    return red_result + green_result


# Manipulate the current Camera frame
def manipulate(cam_to_board):
    global counter
    global dart_thrown
    global coords
    global point_history
    global current_point
    ret, frame = cap.read()

    if ret:
        try:
            cam_to_board = calibrate_dartboard(frame)
            warp = cv2.warpPerspective(frame, cam_to_board, (906, 906))
            warp = circular_mask(warp)
        except:
            warp = frame
            print('no work')

    while cap.isOpened():
        if ret:
            mask = object_detector.apply(warp)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            dart_detection(contours)
            if dart_thrown:
                counter += 1
                if int(counter) > 5:
                    counter = 0
                    dart_thrown = False
                    print(min(coords))
                    if min(coords) == (0, 0):
                        coords = []
                        return
                    x, y = get_point_value(min(coords))
                    current_point = (x, y)
                    point_history.append((min(coords)))
                    coords = []

            if len(point_history) > 0:
                for c in point_history:
                    x, y = c
                    cv2.drawMarker(warp, (x, y), (125, 125, 255), cv2.MARKER_CROSS, 10, 5)
                if len(point_history) == 4:
                    point_history = []

            return warp