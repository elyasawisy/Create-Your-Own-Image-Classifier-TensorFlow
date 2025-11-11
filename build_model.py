import warnings
warnings.filterwarnings('ignore')


import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


(train_set, valid_set, test_set), dataset_info = tfds.load('oxford_flowers102', split= ['train', 'validation', 'test'], as_supervised = True, with_info = True )





total_num_exm = dataset_info.splits['train'].num_examples

num_classes = dataset_info.features['label'].num_classes

print('The Dataset has a total of:')
print(f'\u2022 {total_num_exm:,} images')
print(f'\u2022 {num_classes:,} classes')


for image , label in train_set.take(3):
    print("Image shape:", image.shape)
    print("Label:", label.numpy())
    print("*****")
    
    

batch_size = 32
image_size = 224

num_training_exm = len(train_set)

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255.0
    return image , label

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


training_batches = train_set.shuffle(num_training_exm //4).map(format_image).map(lambda x, y: (data_augmentation(x), y)).batch(batch_size).prefetch(1)
valid_batches = valid_set.map(format_image).batch(batch_size).prefetch(1)
testing_bathces = test_set.map(format_image).batch(batch_size).prefetch(1)


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL,
                                   input_shape = (image_size, image_size, 3),
                                   trainable = False)





model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(102 , activation = 'softmax')
])

model.summary()





optimizers = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer = optimizers , loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
EPOCHS =30


early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
reduceLRO = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6) 

history = model.fit(training_batches,
                    epochs = EPOCHS,
                    validation_data= valid_batches,
                   callbacks = [early_stop, reduceLRO])



training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range=range(EPOCHS)




model.save('flower_classifier.h5')
model.save('flower_classifier_tf', save_format='tf')

