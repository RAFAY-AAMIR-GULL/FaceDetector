import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon=minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def FindFaces(self,img, draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)
        bboxes=[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC=detection.location_data.relative_bounding_box
                ih,iw,ic=img.shape
                bbox=int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([bbox,detection.score])
                if draw:
                    img=self.fancyDraw(img,bbox)
                    cv2.putText(img, f'{int((detection.score[0]*100))}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)


        return img, bboxes

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x,y,w,h=bbox
        x1,y1 = x+w, y+h
        cv2.rectangle(img, bbox, (100, 200, 0), rt)

        # Top Left x,y
        cv2.line(img,(x,y),(x+l,y),(100,200,0),t)
        cv2.line(img, (x, y), (x, y+l), (100, 200, 0), t)

        # Top Right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (100, 200, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (100, 200, 0), t)

        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x + l, y1), (100, 200, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (100, 200, 0), t)

        # Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (100, 200, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (100, 200, 0), t)
        return img
    


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector=FaceDetector()
    while True:
        success, img = cap.read()
        img,bboxes=detector.FindFaces(img)
        print(bboxes)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int((fps))}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)
        cv2.imshow("Image", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


if __name__=="__main__":
    main()
