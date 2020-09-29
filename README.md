# Machine Learning Nanodegree
## Deep Learning
## Project: Image Classifier for Flowers using Tensorflow

----
## Introduction

This project was part of my 'Intro to Machine Learning Nanodegree' provided by Udacity. The code and text in this project is a combination of my own work and that of Udacity.

The goal of this project was to create a tensorflow DNN model that can accurately make predictions on flower images by correctly classifying them. The dataset used to train the network is from the `oxford_flowers102` available on `tensorflow_datasets`, which contains 102 flow categories commonly occuring in the United Kingdom.

This project is divided into two parts; training of the model in the notebook, and converting of the trained model into a command line application using `predict.py`.

1) The notebook contains:

- Importing of the training, test and validation sets
- Training a keras model using the training and validation sets
- Evaluating the model using the test set
- Saving of the trained keras model

2) `predict.py` works by:
- Accepting two positional arguments; the `image_path` of the image wanting to make inference on, and the `saved_model` from the notebook.
- It also accepts two optional arguments; `top_k` representing top **K** labels with the highest probabilities, and `category_names`, a json file that maps the numerical labels to flower names.
- It then returns a table with the top K labels and their probabilities. If top_k was not specified it would only return the label with the highest probability. Using `category_names` would return the flower names instead of the numerical labels.
