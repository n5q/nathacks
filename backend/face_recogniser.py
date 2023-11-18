import cv2
import sys
video = cv2.VideoCapture(0)

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

def process(frame):
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = face_haar_cascade.detectMultiScale(converted, 1.32, 5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0xFF,0), thickness=5)

if __name__ == "__main__":
    while (1):
        ret, frame = video.read()
        processed_frame = process(frame)
        cv2.imshow("frame", frame)


        k = cv2.waitKey(1)
        if (k%256 == 27):
            break
        
        
    video.release()
    cv2.destroyAllWindows()
