import cv2

image_path = 'abba.png'
casc_path = 'haarcascade_frontalface_default.xml'

#to create the cascade(haar cascade)

face_cascade = cv2.CascadeClassifier(casc_path)

# to read the image and transform it to grayscale for normalization
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Now, to detect the faces in abba.png
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (30,30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

# displays found faces and higlights them

print("Found {} faces!".format(len(faces)))

for (x, y, width, height) in faces:
    cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)


