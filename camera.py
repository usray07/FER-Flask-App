import cv2
from model import FacialExpressionModel
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing import image


facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
model = FacialExpressionModel("model.json", "model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        res, frame = self.video.read()
        gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image)
        height, width , channel = frame.shape
        sub_img = frame[0:int(height/6),0:int(width)]
        
        black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
        res = cv2.addWeighted(sub_img, 0.7, black_rect,1, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        FONT_SCALE = 0.8
        FONT_THICKNESS = 2
        try:

            for (x,y, w, h) in faces:
                cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
                roi_gray = gray_image[y:y+h,x-5:x+w+5]
                roi_gray=cv2.resize(roi_gray,(48,48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis = 0)
                image_pixels /= 255
                pred = model.predict_emotion(image_pixels)
                cv2.putText(frame, "{}".format(pred), (x+w+2,y+h+2), FONT,0.7, (240,16,255),2)  

            lable = "Face(s) Detected"
            lable_dimension = cv2.getTextSize(lable,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
            textX = int((res.shape[1] - lable_dimension[0]) / 2)
            textY = int((res.shape[0] + lable_dimension[1]) / 2)
            cv2.putText(res, lable, (textX,textY), FONT, FONT_SCALE, (238,244,21), FONT_THICKNESS)
            cv2.putText(res, " ".format(pred), (0,textY+22+5), FONT,0.7, (238,244,21),2)  
            
  
        except:
            black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
            res = cv2.addWeighted(sub_img, 0.7, black_rect,1, 0)
            lable = "No Face Detected!! "
            lable_dimension = cv2.getTextSize(lable,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
            textX = int((res.shape[1] - lable_dimension[0]) / 2)
            textY = int((res.shape[0] + lable_dimension[1]) / 2)
            
            cv2.putText(res, lable, (textX,textY), FONT, FONT_SCALE, (0,0,250), FONT_THICKNESS)

        frame[0:int(height/6),0:int(width)] =res


        
        res, jpeg = cv2.imencode('.jpg', frame)
        # cv2.waitKey(50)
        return jpeg.tobytes()
