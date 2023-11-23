from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
import os

IMAGE_PATH = r"/content/drive/MyDrive/MAJOR PROJECT 89/MINE11.jpg"
OUTPUT_DIRECTORY = r"/content/drive/MyDrive/MAJOR PROJECT 89/augume11"
image = load_img(IMAGE_PATH)
image = img_to_array(image)
image = np.expand_dims(image, axis=0) 
image1 = cv2.imread(IMAGE_PATH)
#---------------------------------------------------------------------------------------------------------------------
def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

datagen_compound= ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    brightness_range=[0.2,5.0]
)

datagen_compound1= ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=46,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=True,
    validation_split=0.6,
    brightness_range=[0.2,2.0]
)
datagen_compound2=ImageDataGenerator(
    rotation_range =15, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2,  
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip = True, 
    fill_mode = 'nearest', 
    data_format='channels_last', 
    brightness_range=[0.1,0.9]
)
datagen_compound3=ImageDataGenerator(
    rescale=1. / 256,  
    rotation_range=75,  
    width_shift_range=0.3, 
    height_shift_range=0.1,  
    fill_mode='constant',
    cval=0,
    zoom_range=[.9, 1.25],
)

datagen_compound4=ImageDataGenerator(
    rotation_range =50, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2,  
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip = True, 
    fill_mode = 'nearest', 
    data_format='channels_last', 
    preprocessing_function=to_grayscale_then_rgb
)
                         
#---------------------------------------------------------------------------------------------------------------------

seq1 = iaa.Sequential([
    iaa.Fliplr(0.5), 
    iaa.Crop(percent=(0, 0.1)), 
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)
imglist1 = []
img1 = np.asarray( image1 )

seq2 = iaa.Sequential([
    iaa.Fliplr(0.2), 
    iaa.Crop(percent=(0, 0.2)), 
    iaa.Sometimes(
        0.6,
        iaa.GaussianBlur(sigma=(0, 0.2))
    ),
    iaa.LinearContrast((0.35, 2.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.06*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-8, 8)
    )
], random_order=True)

imglist2 = []
img2 = np.asarray( image1 )

seq3 = iaa.Sequential([
    iaa.Fliplr(0.2), 
    iaa.Crop(percent=(0, 0.2)), 
    iaa.Sometimes(
        0.6,
        iaa.GaussianBlur(sigma=(0, 0.2))
    ),
    iaa.GammaContrast((0.5, 2.0)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.06*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-195, 195),
#         shear=(-8, 8)
    )
], random_order=True)

imglist3 = []
img3 = np.asarray( image1 )
#---------------------------------------------------------------------------------------------------------------------

PREFIX = 'img_'
imGen1 = datagen_compound.flow(image, batch_size=1, save_to_dir = OUTPUT_DIRECTORY, 
                    save_prefix=PREFIX, save_format='jpg')
imGen2 = datagen_compound1.flow(image, batch_size=1, save_to_dir = OUTPUT_DIRECTORY, 
                    save_prefix=PREFIX, save_format='jpg')
imGen3 = datagen_compound2.flow(image, batch_size=1, save_to_dir = OUTPUT_DIRECTORY, 
                    save_prefix=PREFIX, save_format='jpg')
imGen4 = datagen_compound3.flow(image, batch_size=1, save_to_dir = OUTPUT_DIRECTORY, 
                    save_prefix=PREFIX, save_format='jpg')
imGen5 = datagen_compound4.flow(image, batch_size=1, save_to_dir = OUTPUT_DIRECTORY, 
                    save_prefix=PREFIX, save_format='jpg')
#---------------------------------------------------------------------------------------------------------------------
for i in range(30):
    images_aug1 = seq1(image=img1)
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY , 'img_{}.jpg'.format(i)), images_aug1) 

    images_aug2 = seq2(image=img2)
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY , 'img_{}.jpg'.format(i+30)), images_aug2)
    
    images_aug3 = seq3(image=img3)
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY , 'img_{}.jpg'.format(i+60)), images_aug3)
#---------------------------------------------------------------------------------------------------------------------

for i in range(30):  
    batch1 = imGen1.next()
    batch2 = imGen2.next()
    batch3 = imGen3.next()
    batch4 = imGen4.next()
    batch5 = imGen5.next()
   