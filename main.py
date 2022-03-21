
from pyexpat import model
import numpy as np
import sys

from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import Flask ,render_template
from flask import jsonify
from flask import request
import cv2 as cv
from flask_cors import CORS

app = Flask(__name__) #app=instance of Flask, __name__ argument is the name of application module
CORS(app)

def get_model():
    global model
    model=load_model('data_augmentation_model_v3.h5')
    print('Model loaded')
def preprocess_image(image,target_size):
    
    image=image.resize(target_size)
    image=img_to_array(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image =np.expand_dims(image,axis=0)
    return image

print("Loading Keras model")
get_model()
@app.route('/sample',methods=['POST'])#decorator to tell url for function underneath to be called
def running():
    
    image=request.files['image']
  
    processed_image=preprocess_image(Image.open(image),target_size=(32,64))
    prediction=model.predict(processed_image)*100
    print(prediction,file=sys.stderr)
    max_value = max(prediction[0])
    output_array = [1 if i >= max_value else 0 for i in prediction[0]]
    z = output_array.index(1)
    response={
        'predicition':{
            'name':z
        }
    }
    return jsonify(response)
    
@app.route('/')
def index():
    return render_template('index.html')
    

    
