#!/usr/bin/env python2

import argparse
import math
import subprocess
import time

import cv2

# Minimum time in seconds to wait between any two actions
FRAME_INTERVAL = 0.05
# Minimum time to wait between detecting faces
DETECT_INTERVAL = FRAME_INTERVAL * 4
# Minimum deviation from 0 degrees in either direction to consider a tilt
TILT_ANGLE = 30
# GUI window name
WINDOW_NAME = 'OpenCV Page Turner'

# These constants were previously included in the OpenCV bindings for Python but
# were removed in a later version.
# https://docs.opencv.org/3.3.0/d9/d31/group__objdetect__c.html
CV_HAAR_FIND_BIGGEST_OBJECT = 4
CV_HAAR_DO_ROUGH_SEARCH = 8


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


def detect_faces(classifier, img, angle):
    return classifier.detectMultiScale(
        rotate_image(img, angle),
        flags=CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH,
        minNeighbors=3,
        minSize=(120, 120),
        scaleFactor=1.3,
    )


def send_linux_keypress(angle):
    key_name = 'Page_Down' if angle > 0 else 'Page_Up'
    subprocess.call(['xdotool', 'key', key_name])
    print('Sent {}'.format(key_name))


def turn_pages(classifier_file, show_ui=True):
    # Initialize OpenCV stuff
    camera = cv2.VideoCapture(0)
    classifier = cv2.CascadeClassifier(classifier_file)
    if show_ui:
        cv2.namedWindow(WINDOW_NAME)

    # Main loop
    last_keypress_time = 0
    last_detect_time = 0
    while True:
        time.sleep(FRAME_INTERVAL)
        now = time.time()

        # Take photo
        _, img = camera.read()
        img = cv2.flip(img, 1)

        # Process photo
        if now - last_detect_time > DETECT_INTERVAL:
            last_detect_time = now
            for angle in [-TILT_ANGLE, TILT_ANGLE]:
                # Locate face
                faces = detect_faces(classifier, img, angle)
                if not len(faces):
                    continue

                # Draw box around face
                if show_ui:
                    x, y, w, h = rotate_point(faces[-1], img, -angle)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Ignore if too soon since last command
                if now - last_keypress_time < 1:
                    continue

                # Send keypress
                send_linux_keypress(angle)
                last_keypress_time = now
                break

        if show_ui:
            cv2.imshow(WINDOW_NAME, img)

            # Exit if program window receives `q` keypress. (In headless mode,
            # must exit with Ctrl+C)
            if cv2.waitKey(5) == 113:
                print('Exiting')
                break

    if show_ui:
        cv2.destroyWindow(WINDOW_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--classifier',
        default='haarcascade_frontalface_alt2.xml',
        dest='classifier_file',
    )
    parser.add_argument('--headless', action='store_true', help='Hide GUI')
    args = parser.parse_args()
    turn_pages(args.classifier_file, show_ui=not args.headless)


if __name__ == '__main__':
    main()
