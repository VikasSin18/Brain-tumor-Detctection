import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt
# import pydotplus
# import graphviz
from io import BytesIO




model=load_model('BrainTumorCatnew3.h5')

image=cv2.imread('C:\\Users\\Vikas Singh B\\OneDrive\\Desktop\\brain tumor using deep learning\\datasets\\meningioma_tumor\\m (6).jpg')
img=Image.fromarray(image)
img=img.resize((150,150))
img=np.array(img)
# print(img)

input_img=np.expand_dims(img,axis=0)
reslt=model.predict(input_img)



#---------for binary-------#
# threshold = 0.5  # You can adjust this threshold as needed
# result = (reslt > threshold).astype(int)
# print(result)


#-----for CAtegorical----#
# p=np.argmax(reslt,axis=1)
# print(p)

#plotting


# Load the saved model
try:
    # Load the saved model
    model = load_model('BrainTumorCatnew3.h5')

    # Plot the model architecture using keras.utils.plot_model
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_plot.png', expand_nested=True, dpi=300)

    # Read the generated plot image
    img = plt.imread('model_plot.png')

    # Display the generated plot using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")




