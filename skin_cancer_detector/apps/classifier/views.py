from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
from PIL import Image
from tensorflow import keras
from ..common.constant import DICT_TARGETS
# Create your views here.
model = None
def index(request):
    if request.method == 'POST':
        image_file = request.FILES['image']        
        input_shape = (128, 128)
        img = Image.open(image_file)
        img = img.convert('RGB')
        img = img.resize(input_shape)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        result = model.predict(img_array)
        argmax_preds = np.argmax(result, axis=1)  # BY ROW, BY EACH SAMPLE
        argmax_preds = keras.utils.to_categorical(argmax_preds)
        prediction = decode_array(argmax_preds)
        return HttpResponse(DICT_TARGETS[prediction[0]])
    else:
        fruit_name = "Upload an image to get the classification"
    
    return render(request, "classifier/index.html", {'fruit_name': fruit_name})

def load_model(request):
    from tensorflow.keras.models import load_model
    global model
    model = load_model('/Users/dhruvrai/Desktop/skin cancer app/skin_cancer_detector/skin_cancer_detector/models/model_resnet.h5')
    return HttpResponse("RESNET34 MODEL LOADED")
    
def decode_array(array):
    try:
        decoded_array = np.argmax(array, axis=1) 
        return decoded_array
    except:
        raise
    