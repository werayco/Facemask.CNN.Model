
# We'd be using Kaggle's API
!pip install kaggle

# so basically, we'd create a folder named '.kaggle' and copy and paste the json file we got from kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d omkargurav/face-mask-dataset

# importing our dependencies
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten,Dropout
from keras.models import Sequential

# Unzipping the files containing the images
from zipfile import ZipFile
path = "/content/face-mask-dataset.zip"

with ZipFile(path,'r') as zip:
  zip.extractall()

# first lets create a label for the data
withmask=os.listdir("/content/data/with_mask")
withoutmask=os.listdir("/content/data/without_mask")

# 1 for images with mask and 0 for without mask
labelsForMask=[1]*len(withmask)
labelsForWithoutMask=[0]*len(withoutmask)
labels = labelsForMask + labelsForWithoutMask

# Image Processing 01
withmaskpath='/content/data/with_mask/'
data = []
for everyimage in withmask:
  image = Image.open(withmaskpath + everyimage)
  image = image.resize((120,120))
  image = image.convert("RGB")
  image = np.array(image)
  data.append(image)

# Image Processing 02
withoutmaskpath='/content/data/without_mask/'
for every_image in withoutmask:
  # the PIL.Image helps us open the image file and returns an image object, unlike OpenCV, which returns an array of intensity values of the pixels
  image = Image.open(withoutmaskpath + every_image)
  image = image.resize((120,120))
  image = image.convert("RGB")
  image = np.array(image)
  data.append(image)

# converting both the images and the labels to array respectively
x = np.array(data)
y = np.array(labels)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

# scaling the images
x_train_scaled=x_train/255
x_test_scaled=x_test/255

# neural network session
model= Sequential([
    Conv2D(filters=32,kernel_size=(3,3), activation='relu',input_shape=(120,120,3),strides=(1,1)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64,kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(2,activation='sigmoid')
])

# configuring our model
model.compile(optimizer="Adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

# fitting our model to the data
fitter = model.fit(x_train_scaled,y_train,validation_split=0.1,epochs=5)

loss,accuracy =model.evaluate(x_test_scaled,y_test)
print("test accuracy: ",accuracy)

# Visualization section
plt.plot(fitter.history["loss"],label="train_loss")
plt.plot(fitter.history["val_loss"],label="vald loss")
plt.legend()
plt.show()

plt.plot(fitter.history["accuracy"],label="train_acc")
plt.plot(fitter.history["val_accuracy"],label="vald acc")
plt.legend()
plt.show()

import pickle as plk
pathfordata=input("insert the path of your data: ")
image=cv2.imread(pathfordata)
image_resize= cv2.resize(image,(120,120))
image_scaled = image_resize/255
image_reshape=np.reshape(image_scaled,[1,120,120,3])
pred = model.predict(image_reshape)
print(pred)

inp_label=np.argmax(pred)
print(inp_label)

if inp_label == 0:
  print("the person isnt wearing a mask")

else:
  print("the person is wearing a mask")

import pickle as pkl
with open("facemaskmodel.pkl","wb") as FMM:
  pkl.dump(model,FMM)
