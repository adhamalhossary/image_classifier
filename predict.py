import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import json

import argparse

keras_model = tf.keras.models.load_model('flower_image_classifier.h5', custom_objects={'KerasLayer':hub.KerasLayer})

keras_model.summary()

with open('label_map.json', 'r') as f:
    class_names = json.load(f)
    class_names = {int(key): class_names[key] for key in class_names}

def process_image(image):

  tensor_image = tf.image.resize(image,(224,224))
  tensor_image /= 255
  numpy_image = tensor_image.numpy()

  return numpy_image


def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)

    predictions = model.predict(expanded_test_image)

    top_k_values, top_k_indices = tf.nn.top_k(predictions, k=top_k)

    top_k_values = top_k_values.numpy()
    top_k_labels = (top_k_indices + 1).numpy()

    return top_k_values, top_k_labels, processed_test_image




#
# import os
#
# directory = './test_images'


# for filename in os.listdir(directory):
#     if filename.endswith(".jpg"):
#       image_path = os.path.join(directory, filename)
#
#       probs, labels, image = predict(image_path, keras_model, 5)
#
#       class_labels = [class_names.get(label) for label in labels[0]]
#
#       fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
#       ax1.imshow(image)
#       ax1.axis('off')
#       ax1.set_title(filename.strip('.jpg').replace('_'," "))
#
#       ax2.barh(np.arange(5), probs[0])
#       ax2.set_aspect(0.1)
#       ax2.set_yticks(np.arange(5))
#       ax2.set_yticklabels(class_labels, size='large');
#       ax2.set_title('Class Probability')
#       ax2.set_xlim(0, 1.1)
#       plt.tight_layout()
#
#       plt.show()