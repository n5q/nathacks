import cv2
import sys
video = cv2.VideoCapture(0)


if __name__ == "__main__":
    while (1):
        ret, frame = video.read()
        cv2.imshow("frame", frame)
    video.release()
    cv2.destroyAllWindows()
