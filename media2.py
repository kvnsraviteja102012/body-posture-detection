#for finding the pose structure of the person
import mediapipe as mp
import cv2 as v
import time as t
cap = v.VideoCapture(0)
obj = mp.solutions.pose
pose = obj.Pose()
draw = mp.solutions.drawing_utils
ctimme = 0
ptime = 0
while True:
    res, frame = cap.read()
    img = v.cvtColor(frame, v.COLOR_BGR2RGB)
    # processing of the image with for finding corodinates
    results = pose.process(img)
    if results.pose_landmarks:
        draw.draw_landmarks(frame, results.pose_landmarks,
                            obj.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            # exact length of the pixels in the immage
            cx = int(lm.x*w)
            cy = int(lm.y*h)
            if id == 20:
                v.circle(frame, (cx, cy), 10, (255, 0, 255), 6)
    ctime = t.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    font = v.FONT_HERSHEY_SIMPLEX
    v.putText(frame, str(int(fps)), (10, 50), font, 3, (255, 0, 0))
    v.imshow('image', v.flip(frame, flipCode=1))
    k = v.waitKey(1) & 0XFF
    if k == ord('s'):
        break
v.destroyAllWindows()
