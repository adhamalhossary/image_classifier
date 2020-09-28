import tensorflow as tf

import numpy as np
from PIL import Image

# This function is used to resize and normalize the image for appropriate usage by the model for accurate predictions
def process_image(image):
    tensor_image = tf.image.resize(image, (224, 224))
    tensor_image /= 255
    numpy_image = tensor_image.numpy()

    return numpy_image

# This function is used to return two lists; one containing the labels with the highest probabilities, and the other
# containing the probabilities

def predict(image_path, model, top_k=1):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)

    predictions = model.predict(expanded_test_image)

    top_k_values, top_k_indices = tf.nn.top_k(predictions, k=top_k)

    top_k_values = top_k_values.numpy()
    top_k_labels = (top_k_indices + 1).numpy()

    return top_k_values, top_k_labels
