import cv2

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error loading face cascade.")
else:
    print("Face cascade loaded successfully.")

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = gray[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(gray_roi, scaleFactor=1.8, minNeighbors=20)
        
        for (x1, y1, w1, h1) in smiles:
            cv2.rectangle(face_roi, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)

            # Save the frame only when a smile is detected
            cv2.imwrite('selfie.png', frame)

    cv2.imshow('cam star', frame)
    
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()

