import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.utils import plot_model
from io import BytesIO
import base64

app = Flask(__name__)


model =load_model('BrainTumorCatnew3.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return """Yes Brain Tumor\n
                  TYPE-Glioma Tumour"""
    elif classNo == 2:
        return "Yes Brain Tumor\n TYPE-Meningioma Tumor"
    elif classNo == 3:
        return "Yes Brain Tumor\n TYPE-Pituitary Tumor"


      


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((150, 150))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    p=np.argmax(result,axis=1)
    return p


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



def load_and_plot_model():
    # Load the pre-trained model
    model = load_model('BrainTumorCatnew3.h5')

    # Plot the model architecture
    plot_image_stream = BytesIO()
    plot_model(model, to_file=plot_image_stream, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)
    plot_image_stream.seek(0)
    encoded_plot_image = base64.b64encode(plot_image_stream.read()).decode('utf-8')

    return encoded_plot_image

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


#graph plotting



@app.route('/', methods=['GET'])
def index1():
    # Generate the plot
    plot_image = load_and_plot_model()

    # Render the HTML template with the plot image
    return render_template('index.html', plot_image=plot_image)



if __name__ == '__main__':
    app.run(debug=True)