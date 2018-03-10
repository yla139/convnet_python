import os, shutil
import keras
import matplotlib.pyplot as plt
# This is module with image preprocessing utilities
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir ='C:\\Users\\MINE\\Desktop\\Harvard Extension\\Deep Learning\\Assignments\\assi6\\train'

# The directory where we will
# store our smaller dataset
base_dir = 'C:\\Users\\MINE\\Desktop\\Harvard Extension\\Deep Learning\\Assignments\\assi6\\data_problem2'
os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)


# train_dogs_dir
fnames = [os.path.join(train_dogs_dir, fname) for fname in os.listdir(train_dogs_dir)]

# We pick one image to "augment"
img_path = fnames[10]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# Note that the validation data should not be augmented!
train_datagen_horizontal_flip = ImageDataGenerator(horizontal_flip=True)
train_datagen_rotation_range = ImageDataGenerator(rotation_range=40)
train_datagen_width_shift = ImageDataGenerator(width_shift_range=0.5)
train_datagen_shear_range = ImageDataGenerator(shear_range=0.2)
test_datagen_zoom_range = ImageDataGenerator(zoom_range=0.5)

# make directory to save pictures
save_dir = os.path.join(base_dir, 'pictures')
os.mkdir(save_dir)

file_name = 'figure_'

i = 0
for batch in train_datagen_horizontal_flip.flow(x, save_to_dir=save_dir, save_prefix=file_name+str(i), batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    break

for batch in train_datagen_rotation_range.flow(x, save_to_dir=save_dir, save_prefix=file_name+str(i), batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    break

for batch in train_datagen_width_shift.flow(x, save_to_dir=save_dir, save_prefix=file_name+str(i), batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    break

for batch in train_datagen_shear_range.flow(x, save_to_dir=save_dir, save_prefix=file_name+str(i), batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    break

for batch in train_datagen_zoom_range.flow(x, save_to_dir=save_dir, save_prefix=file_name+str(i), batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    break
plt.show()



