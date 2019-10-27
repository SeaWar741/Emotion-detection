import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye_tree_eyeglasses.xml')
profile_cascade = cv2.CascadeClassifier('./haarcascade_profileface.xml')
smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')
hand_cascade = cv2.CascadeClassifier('./palm.xml')
fist_cascade = cv2.CascadeClassifier('./fist.xml')
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = eye_cascade.detectMultiScale(gray,1.1,4)
    profiles = profile_cascade.detectMultiScale(gray,1.1,4)
    smiles = smile_cascade.detectMultiScale(gray,1.1,100)
    hands = hand_cascade.detectMultiScale(gray,1.1,1)
    fists = fist_cascade.detectMultiScale(gray,1.1,1)
    # Draw the rectangle around each face
    emotion = ""
    if(len(smiles)>0):
        emotion = " Smiling"
        if(len(fists)>0 or len(hands)>0):
            emotion= emotion+ " Waving"
    else:
        if(len(fists)>0 or len(hands)>0):
            emotion= " Waving"
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img,"Person"+emotion,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.0,(255, 255, 255),lineType=cv2.LINE_AA)
    for (x,y,w,h) in profiles:
        cv2.rectangle(img,(x,y),(x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img,"Person",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.0,(255, 255, 255),lineType=cv2.LINE_AA)

    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 128, 0), 2)
    for (x, y, w, h) in smiles:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,155), 2)
        #cv2.putText(img,"Smiling",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.0,(0,0,155),lineType=cv2.LINE_AA)
        emotion = " Smiling"
    for (x,y,w,h) in hands:
        #cv2.rectangle(img,(x,y),(x+w, y+h), (255,255,255),2)
        cv2.putText(img,"Hand",(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),lineType=cv2.LINE_AA)
    for (x,y,w,h) in hands:
        #cv2.rectangle(img,(x,y),(x+w, y+h), (255,255,255),2)
        cv2.putText(img,"Hand",(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),lineType=cv2.LINE_AA)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()