import cv2
import numpy as np
import keras
video = cv2.VideoCapture(0)

# EMOTIONS = (
#     'angry',
#     'disgust',
#     'fear',
#     'happy',
#     'sad',
#     'surprise',
#     'neutral'
# )
EMOTIONS = ('Angry','Happy','Neutral','Sad','Surprise')

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

model = keras.models.load_model("EmotionDetectionModel.h5")

def get_emotion(face) -> str:
    face = cv2.resize(face, (48,48), interpolation=cv2.INTER_AREA)
    pixels = keras.preprocessing.image.img_to_array(face)
    pixels = np.expand_dims(pixels, axis=0)/255
    prediction = model.predict(pixels)
    i = np.argmax(prediction[0])
    emotion = EMOTIONS[i]
    return emotion
    

def process(frame) -> np.ndarray:
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_haar_cascade.detectMultiScale(converted, 1.32, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0xFF,0), thickness=2)
        face = converted[y:y+w, x:x+h]
        emotion = get_emotion(face)
        cv2.putText(frame, emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0xFF, 0), 2)
    return faces
        
        
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