import cv2
import matplotlib.pyplot as plt
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

cap = cv2.VideoCapture('data/2.mp4')

prev_frame_time = time.time()
new_frame_time = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print('Finished processing')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x  + w, y + h), (255, 0, 0), 4)
        cv2.putText(frame, 'Detection', (x + 75, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
    
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps_text = f'FPS: {fps}'

    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Face Detection in Video', frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()