import os, shutil
import keras
import matplotlib.pyplot as plt
# This is module with image preprocessing utilities
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import regularizers
from keras import optimizers

# The path to train data and validation data
train_dir = 'C:\\Users\\MINE\\Desktop\\Harvard Extension\\Deep Learning\\Assignments\\assi6\\data_sample\\train'
validation_dir = 'C:\\Users\\MINE\\Desktop\\Harvard Extension\\Deep Learning\\Assignments\\assi6\\data_sample\\validation'

# build model
print("============== build model =================")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))    # same lambda value as q1


# compile model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# training
print("============== training begins =============== ")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)


# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# training 
print("============== start fitting =============== ")
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,           # half of the epochs
      validation_data=validation_generator,
      validation_steps=50)

# save model
model.save('HW5P3.h5')

print("============== plotting graphs =============== ")
# compute graph accuracy with the combined regularization and augmentation
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
