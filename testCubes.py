import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

dataPath = "C:/Users/janni/Desktop/ml/best"

# parameters
batch_size = 32
img_height = 224
img_width = 224
lr = 1e-4

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataPath,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataPath,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
class_no = len(class_names)

model = keras.models.load_model('model/backup')

for test in os.listdir("Resources/testData"):

    # testing
    testedImage = "Resources/testData/" + test
    img = keras.preprocessing.image.load_img(
        testedImage, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # printing results
    print(
        "\nImage {} classified as {} with a {:.2f}% confidence."
            .format(testedImage, class_names[np.argmax(score)], 100 * np.max(score)))
    print("All classifications : ")
    for index in range(class_no):
        print("\t As {} with {:.2f}% confidence"
              .format(class_names[index], 100 * score[index]))