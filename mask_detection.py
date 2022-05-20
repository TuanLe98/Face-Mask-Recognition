import os
import cv2
import datetime
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_DIR)

class FaceMask:
    face_detect = None
    model = None

    def __init__(self):
        self.face_detect = cv2.CascadeClassifier(os.path.join(CURRENT_DIR,"haarcascade_frontalface_default.xml"))
        if(self.face_detect.empty()):
            print("cascade empty")

        modelfile = os.path.join(CURRENT_DIR,"face_mask.h5")
        if(os.path.exists(modelfile)):
            self.model = load_model(modelfile)
        np.set_printoptions(suppress=True)

    def gray_image(self, frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = self.face_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)
        return faces

    def mask_predict(self, images):
        test_image = image.load_img(images,target_size=(128,128,3))
        test_image = image.img_to_array(test_image)
        test_image = np.array(test_image).reshape(-1,128,128,3)
        predict = self.model.predict(test_image)
        return predict

    def detectFaceInFrame(self, frame):
        for (x,y,w,h) in self.gray_image(frame):
            face_img = frame[y:y+h,x:x+w]
            cv2.imwrite('facemask.jpg',face_img)
            predict = self.mask_predict('facemask.jpg')
            print(predict)
            if predict == 1:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255),3)
                cv2.putText(frame,"No Face Mask", ((x + w) // 2,y + h + 30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, "Face Mask", ((x + w)//2, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
            date = str(datetime.datetime.now())
            cv2.putText(frame,date,(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

faceMask = FaceMask()

def run():
    cap = cv2.VideoCapture(os.path.join(CURRENT_DIR,'mask.mp4'))
    while True:
        ret, frame = cap.read()
        faceMask.detectFaceInFrame(frame)
        cv2.imshow('Face Mask', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()