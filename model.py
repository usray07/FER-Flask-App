from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow 

# config = tensorflow.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.15
# session = tensorflow.compat.v1.Session(config=config)


class FacialExpressionModel(object):
    EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    def __init__(self,model_json_file, model_weights_file):
        self.loaded_model = model_from_json(open(model_json_file, "r").read())
        self.loaded_model.load_weights(model_weights_file)
        # self.loaded_model._make_predict_function()
    
                  
    def predict_emotion(self,img):
        self.preds = self.loaded_model.predict(img)
        max_index = np.argmax(self.preds[0])
        # print(self.preds)
        # print(max_index)
        # print(FacialExpressionModel.EMOTIONS_LIST[max_index])
        return FacialExpressionModel.EMOTIONS_LIST[max_index]
                  
                  