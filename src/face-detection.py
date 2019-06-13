import cv2
import matplotlib.pyplot as plt



# getting a camera object
cam = cv2.VideoCapture(0)

# cascade classifiers
face_cascade = cv2.CascadeClassifier("../dataset/haarcascades/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("../dataset/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier("../dataset/haarcascades/haarcascade_smile.xml")


# function to detect the face, eye and smile
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_section = frame

    for (x, y, w, h) in faces:
        # face section which is useful for training data preparation
        face_section = frame[y - 10:y + h + 10, x - 10:x + w + 10]
        face_section = cv2.resize(face_section, (100, 100))

        center = (x + w//2, y + h//2)
        cv2.ellipse(frame, center, (w//2 + 10, h//2 + 60), 0, 0, 360, (0, 0, 255), 2)
        roi_color = frame[y:y+h,x:x+w]
        roi_gray = gray[y:y+h,x:x+w]

        # cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = frame[y:y + h, x:x + w]

        # detecting eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (x2, y2, w2, h2) in eyes:
            # eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            # radius = int(round((w2 + h2) * 0.25))
            # cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
            eye_center = (x2 + w2//2, y2 + h2//2)
            cv2.ellipse(roi_color, eye_center, (w2 // 2 + 10, h2 // 2 - 5), 0, 0, 360, (255, 0, 0), 2)

        # detection the lip and smile some says so
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 0), 2)

    return face_section


# video stream
while True:
    ret, frame = cam.read()
    # if camera resource is not available wait and check again
    if not ret:
        continue

    # q is pressed on frame then quit the window
    key_pressed = cv2.waitKey(1) & 0xFF  # bit masking to get last 8-bits
    if key_pressed == ord('q'):
        break

    face_section = detect(frame)
    print(type(face_section))
    print(face_section.shape)


    # now show the frame
    cv2.imshow("Video Frame", frame)



# release the camera after quitting the window

cv2.release()
cv2.destroyAllWindows()

