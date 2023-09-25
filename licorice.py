import cv2
from imutils import face_utils
import dlib

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(1)

def moveDown(myPoints):
    newP = list(myPoints)
    newP[0] = list(newP[0])
    newP[1] = list(newP[1])
    newP[0][1] += 3
    newP[1][1] += 3
    newP[0] = tuple(newP[0])
    newP[1] = tuple(newP[1])
    newP = tuple(newP)
    return newP

start_point = (240,0)
end_point = (255,25)
myPoints = (start_point, end_point, "a")

while True:
    _, image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(face_utils.FACIAL_LANDMARKS_IDXS)

    rects = detector(gray, 0)
    arr = []
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # print(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        arr = shape[60:]
    
    if (len(arr) >= 8):
        # print("in")
        if (abs(arr[3][1] - arr[7][1]) < 10):
            print("closed")
        else:
            print("opened")
    myPoints = moveDown(myPoints)
    color = (0, 0, 0)
    cv2.rectangle(image, myPoints[0], myPoints[1], color, -1)

    cv2.imshow("Output", image)
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cv2.destroyAllWindows()
cap.release()

