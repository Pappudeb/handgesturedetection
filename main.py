import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands= mp.solutions.hands
hands= mpHands.Hands()
mpDraw =  mp.solutions.drawing_utils


#Frame per second

cTime=0
pTime=0




while True:
    ret,img=cap.read()
    imgBGR= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results= hands.process(imgBGR)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),2)


    cv2.imshow("Image", img)
    if cv2.waitKey(1)==13:
       break

cv2.destroyAllWindows()
