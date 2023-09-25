import cv2
from imutils import face_utils
import dlib
import random
import numpy as np

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

def catch(myPoints, mouth, open):
    if (mouth[1][0] + 10 >= myPoints[0][0] and mouth[5][0] - 10 <= myPoints[1][0]) and (myPoints[1][1] + 4 >= mouth[5][1] and myPoints[0][1] - 8 <= mouth[5][1]) and open:
        print("caught")
        return True 
    return False


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

def generateGummyBear():
    spx = random.randrange(100, 520)
    sp = (spx, 0)
    ep = (spx + 20, 30)
    jp = "a"
    myPoints = (sp, ep, jp)
    return myPoints

start_point = (240,0)
end_point = (255,25)
myPoints = (start_point, end_point, "a")
count = 0
points = 0
gummies = []
open = False
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
        #print("in")
        if (abs(arr[3][1] - arr[7][1]) < 10):
            open = False
            # print("closed")
        else:
            open = True
            # print("opened")
    count += 1
    lose = False
    if (count % 40 == 0):
        myPoints = generateGummyBear()
        gummies.append(myPoints)
        color = (0, 0, 0)

    gummies2 = []
    for gummy in gummies:
        gummy = moveDown(gummy)
        add = True
        if (len(arr) >= 8):
            # print("a")
            caught = catch(gummy, arr, open)
            if caught:
                add = False
                points += 1
        if gummy[0][1] <= 480 and add:   
            gummies2.append(gummy)
        elif gummy[0][1] >= 480:
            lose = True
            break
        cv2.rectangle(image, gummy[0], gummy[1], color, -1)
    
    pointText = "Points: " + str(points)
    image = cv2.flip(image, 1)
    cv2.putText(image, pointText, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    gummies = gummies2

    cv2.imshow("Output", image)
    if lose:
        # print("lost")
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if ret:
            while(True):
                (B, G, R) = cv2.split(frame)
                zeros = np.zeros(frame.shape[:2], dtype = "uint8")
                
                cv2.putText(frame, "GAME OVER. TOTAL SCORE: " + str(points), (75, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow("frame", cv2.merge([zeros, zeros, R]))
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cv2.destroyAllWindows()
cap.release()

