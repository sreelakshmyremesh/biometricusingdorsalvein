# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:55:39 2021

@author: USER
"""

IMG_I = 15
IMG_H = 150
CHAl1NELS = 3

INPUT_SHAPE = (IMG_l•I, IMG_H, CHANNELS} NS_CLASSES = 2
EPOCHS = 10
BATCH_SIZE = 6
model = Sequential()
model.add (Corv20(32, (3, 3), input_shape=INPUT_SHAPE}} model.add (Activation('relu '))
model,add (MaxPooling2D(pool_size=(2, 2)))

model.add (Corv2D(32, (3, 3})) model .add (Activation('relu '))
model .add (MaxPooling2D(pool_size=(2, 2)))


model .add (Corv20(64,(3,3})) model.add (Activation("relu")) model.add (Corv20(250,(3,3))}
model .add (Activation("relu"))

model.add (Corv20(128,(3,3))) model .add (Activation("relu")) model .add (AvgPool20(2,2}}

                                                                                    

mode1.add (AvgPoo12D(2,2)) model.add (Conv2D(64,(3,3})) model.add (Activation("relu")) model .add (AvgPool2D(2,2}}

modcl.odd  (Conv2D(256,(2,2))) model .add (Activation("relu")) model.add (MaxPool2D(2,2}}

model.add (Flatten()) model .add (Dense(32)) model.add (Dropout(0.25)} model.add (Dense(1))
model .add (Activation("sigmoid"}) model.compile(loss= 'binary_c rossentropy ',
optimizer= 'rmsprop ', metrics=L•accuracy 'j)

mode1.summary()
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
zoom_range=0.2, horizontal_flip=True, val1dat1on_spl1t=0.3)

train_generator = train_datagen .flow_from_directory( DATASET_DIR ,
target_size= (IMG_H, IMG_ )


                                                                    