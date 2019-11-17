#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install tensorflow')


# In[5]:


from __future__ import absolute_import, division, print_function, unicode_literals
import os, os.path, PIL
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
DIR = './data/Cyrillic'
chars=os.listdir(DIR)
count=dict()
for char in chars:
    count.update({char:len(os.listdir(DIR+'/'+char))})
print(count)


# In[6]:


from skimage import io
from skimage import img_as_float
from skimage.transform import resize
from PIL import Image

img_collection_arr=dict()
img_collection=dict()
for key,value in count.items():
    folder=os.path.join('./data/Cyrillic/',key)
    allfiles=os.listdir(folder)
    imlist=[filename for filename in allfiles if filename[-4:] in [".png",".PNG"]]
    # Assuming all images are the same size, get dimensions of first image
    for image in imlist:
        filename=os.path.join(folder,image)
        camera = io.imread(filename,-1)
        image_file = Image.open(filename) # opens image  
        img = image_file.convert('LA').resize((28,28)) # converts to grayscale w/ alpha
        image_1=np.array(img)
#         print(image_1.shape)
#         MPL.imshow(img)
#         MPL.show()
        if(key not in img_collection_arr.keys()):
            img_collection_arr.update({key:[image_1]})
        else:
            if(len(img_collection_arr[key])<200):
                img_collection_arr[key].append(image_1)
    print(img_collection_arr.keys())


# In[12]:


count=dict()
for key,val in img_collection_arr.items():
    count.update({key:len(val)})
print(count)


# In[7]:


import matplotlib.pyplot as MPL
from sklearn.model_selection import train_test_split
X=[];y=[]
class_names = list(img_collection_arr.keys())
for key,imgs in img_collection_arr.items():
    [X.append(img) for img in imgs]
    [y.append([class_names.index(key)]) for img in imgs]
#     for img in imgs:
#         print(img.shape)
#     print(img)
#     MPL.imshow(img)
#     MPL.show()
train_images, test_images, train_labels, test_labels = train_test_split(np.array(X), np.array(y), 
                                                    test_size=0.2, 
                                                    random_state=0)


# In[21]:


print('classes:',class_names)
print('number of category:',len(class_names))
print("train image collection:",type(train_images),
      "\ntrain image:",type(train_images[0]),
      '\ntraining label collection:',type(train_labels),
      '\ntraining label:',type(train_labels[0]))
print('shape of training image:',train_images[0].shape)


# In[9]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 2)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names), activation='softmax'))
model.summary()


# In[10]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# In[11]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("test accuracy:",test_acc)


# In[ ]:





# In[ ]:





# In[ ]:




