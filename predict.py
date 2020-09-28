import tensorflow as tf
import tensorflow_hub as hub

import json
import argparse

from prettytable import PrettyTable

from supporting_functions import predict

# Argument Parser is used to retrieve user input from command line
parser = argparse.ArgumentParser()

parser.add_argument('image_path', action="store")
parser.add_argument('saved_model', action="store")
parser.add_argument('--top_k', action="store", type=int)
parser.add_argument('--category_names', action="store")

user_input = parser.parse_args()

saved_model = user_input.saved_model
image_path = user_input.image_path
top_k = user_input.top_k
category_names = user_input.category_names

# Keras model is loaded to be used for predictions
keras_model = tf.keras.models.load_model(saved_model, custom_objects={'KerasLayer': hub.KerasLayer})

# If statement is used to detect user input for top_k
if top_k is None:
    top_values, top_labels = predict(image_path, keras_model)
else:
    top_values, top_labels = predict(image_path, keras_model, top_k=top_k)

# top_values and top_labels are indexed to retrieve list of values
top_values = top_values[0]
top_labels = top_labels[0]

# If statement is used to detect user input for category_names
if category_names is None:
    pass
else:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
        class_names = {int(key): class_names[key] for key in class_names}

    top_labels = [class_names.get(label) for label in top_labels]

# PrettyTable is used to print out a neat table of classes with their corresponding probabilities
table = PrettyTable(['Class', 'Probability'])

for value, label in zip(top_values, top_labels):
    table.add_row([label, round(value, 2)])

print(table)
