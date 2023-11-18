import cv2
import sys
video = cv2.VideoCapture(0)


if __name__ == "__main__":
    while (1):
        ret, frame = video.read()
        cv2.imshow("frame", frame)

        k = cv2.waitKey(1)
        if (k%256 == 27):
            break
        
        
    video.release()
    cv2.destroyAllWindows()
