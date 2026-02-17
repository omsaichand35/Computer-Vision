import cv2 as cv
import mediapipe as mp
import time


mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection()

mpDraw = mp.solutions.drawing_utils

capture = cv.VideoCapture(0)

fTime = 0

while True:
    isTrue, frame = capture.read()
    frame = cv.flip(frame, 1)

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = face.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(frame, detection)

            ih, iw, ic = frame.shape

            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            
            cv.rectangle(frame, bbox, (255, 0, 0), 1)

    cTime = time.time()
    fps = 1/(cTime - fTime)
    fTime = cTime

    cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2)

    cv.imshow('Face Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()