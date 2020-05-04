import cv2


casc_path = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(casc_path)

video_capture  = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    adjusted = cv2.convertScaleAbs(frame, alpha = 1.5, beta = 50)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)
        # cv2.rectangle(adjusted, (x, y), (x+width, y+height), (255, 0, 0), 2)
        # cv2.rectangle(gray, (x, y), (x+width, y+height), (255, 0, 0), 2)

    cv2.imshow('Video', frame)
    # cv2.imshow('Video', adjusted)
    # cv2.imshow('Video', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
