import cv2 
from fer import FER 
import numpy as np
import os
images = {
    "nuetral": cv2.imread("nuetral.jpeg", cv2.IMREAD_UNCHANGED),
    "happy": cv2.imread("Emoji.webp", cv2.IMREAD_UNCHANGED),
    "sad": cv2.imread("sadEmoji.webp", cv2.IMREAD_UNCHANGED),
    "angry": cv2.imread("Angry.png", cv2.IMREAD_UNCHANGED),
    "surprise": cv2.imread("Suprise.jpeg", cv2.IMREAD_UNCHANGED),
    "fear" : cv2.imread("fear.png", cv2.IMREAD_UNCHANGED)
    
}

front_tranning_date = cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")

webcam = cv2.VideoCapture(0)

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1900)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

emotionTracker = FER(mtcnn=True)

currentEmotion = ""
currentScore = 0

while True:
    exsist, frame = webcam.read()
    grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cordinates = front_tranning_date.detectMultiScale(grayScaleImage)

    for (x,y,w,h) in cordinates:
        roi = frame[y:y+h, x:x+w]
        emotion, score = emotionTracker.top_emotion(roi)
        if emotion in images.keys() and (currentEmotion!=emotion and score>0.5):
            currentEmotion = emotion
            currentScore = score
            
        if currentEmotion:
            emoji = images.get(currentEmotion)
            emoji = cv2.resize(emoji, (w,h))
            rgb = emoji[:,:,:3]
            alpha = emoji[:,:,3]/255.0
            roi = roi.astype(float)
            rgb = rgb.astype(float)
            alpha = cv2.merge([alpha, alpha, alpha])
            belndedImage = ((1 - alpha) * roi + rgb * alpha)
            frame[y:y+h, x:x+w] = belndedImage
            cv2.putText(frame, f'{currentEmotion}: {currentScore}', (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    
       
    cv2.imshow("PlaceHolder", frame)
    
    cv2.waitKey(1)
    if cv2.getWindowProperty("PlaceHolder", cv2.WND_PROP_VISIBLE)<1:
        break

webcam.release()  
cv2.destroyAllWindows()



    



