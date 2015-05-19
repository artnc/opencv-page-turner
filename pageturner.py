"""
Usage:
    python pageturner.py
"""
import argparse
import cv2
import math
import subprocess
import time

TILT_ANGLE = 30
WINDOW_NAME = 'OpenCV Page Turner'

def rotate_image(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 0.9)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)

def rotate_point(pos, img, angle):
    angle = math.radians(angle)
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)

    x = pos[0] - img.shape[1] * 0.4
    y = pos[1] - img.shape[0] * 0.4

    new_x = x * cos_angle + y * sin_angle + img.shape[1] * 0.4
    new_y = -x * sin_angle + y * cos_angle + img.shape[0] * 0.4
    return int(new_x), int(new_y), pos[2], pos[3]

def detect_face(classifier, img, angle):
    faces = classifier.detectMultiScale(
        rotate_image(img, angle),
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(120, 120),
        flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV_HAAR_DO_ROUGH_SEARCH
    )
    return rotate_point(faces[-1], img, -angle) if len(faces) else None

def send_linux_keypress(angle):
    subprocess.Popen([
        'xdotool',
        'key', 'Page_Down' if angle > 0 else 'Page_Up'
    ]).communicate()

def turn_pages(classifier_file):
    # Create window
    cv2.namedWindow(WINDOW_NAME)

    camera = cv2.VideoCapture(0)
    classifier = cv2.CascadeClassifier(classifier_file)
    last_keypress_time = 0

    while True:
        _, img = camera.read()
        img = cv2.flip(img, 1)

        for angle in [-TILT_ANGLE, TILT_ANGLE]:
            # Locate face
            face = detect_face(classifier, img, angle)
            if not face:
                continue

            # Draw box around face
            x, y, w, h = face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ignore if too soon since last command
            now = time.time()
            if now - last_keypress_time < 1:
                continue

            # Send keypress
            send_linux_keypress(angle)
            last_keypress_time = now

        cv2.imshow(WINDOW_NAME, img)

        # Exit on keypress of program window
        pressed_key = cv2.waitKey(5)
        if pressed_key != -1 and pressed_key not in [65365, 65366]:
            break

    cv2.destroyWindow(WINDOW_NAME)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--classifier',
        dest='classifier_file',
        default='haarcascade_frontalface_alt2.xml'
    )
    args = parser.parse_args()
    turn_pages(args.classifier_file)

if __name__ == '__main__':
    main()
