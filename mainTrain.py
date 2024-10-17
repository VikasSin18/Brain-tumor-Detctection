import cv2
import os
import numpy as np 
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical



image_directory ='datasets/'

no_tumor_images=os.listdir(image_directory+ 'no_tumor/')
yes_glioma_tumor=os.listdir(image_directory+ 'glioma_tumor/')
yes_meningioma_tumor=os.listdir(image_directory+ 'meningioma_tumor/')
yes_pituitary_tumor=os.listdir(image_directory+ 'pituitary_tumor/')
dataset=[]
label=[]

INPUT_SIZE=150
# print(no_tumor_images)

# path='no0.jpg'

# print(path.split('.')[1])

for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no_tumor/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_glioma_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'glioma_tumor/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in enumerate(yes_meningioma_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'meningioma_tumor/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(2)

for i , image_name in enumerate(yes_pituitary_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'pituitary_tumor/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(3)


datasets=np.array(dataset)
label=np.array(label)

x_train,x_test,y_train,y_test=train_test_split(datasets,label,test_size=0.1,random_state=101)
# print('Training set:', x_train.shape)

# print(len(datasets))
# print('------------------------------')
# print(len(label))

x_train=normalize(x_train,axis=2)
x_test=normalize(x_test,axis=2)

y_train=to_categorical(y_train,num_classes=4)
y_test=to_categorical(y_test,num_classes=4)





model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(4))
model.add(Activation('softmax'))

# model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,validation_split=0.1)

model.save('BrainTumorCatnew3.h5')



 