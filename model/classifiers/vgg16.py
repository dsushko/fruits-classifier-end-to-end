import os
import numpy as np
import pandas as pd
from keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from model.classifiers.config import KerasNetworkConfig

from sklearn.utils import shuffle

from utils.eligiblemodules import EligibleModules
from utils.globalparams import GlobalParams

MODEL_SAVE_PATH = os.path.join(GlobalParams().data_path, 'saved_models/vgg16/')

@EligibleModules.register_classifier
class VGG16Classifier:
    
    def __init__(self, **kwargs):
        for param, val in KerasNetworkConfig.parse_obj(kwargs):
            setattr(self, param, val)
        self.classes = GlobalParams().num_classes
        if not os.path.exists(os.path.join(MODEL_SAVE_PATH, 'vgg16.json')):
            self.download_network()
        
        self.model = self.set_up_model()


    def download_network(self):
        network = tf.keras.applications.VGG16(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(self.image_size, self.image_size, 3),
                pooling=None,
                classes=self.classes,
                classifier_activation="softmax",
            )
        model_json = network.to_json()
        with open(os.path.join(MODEL_SAVE_PATH, 'vgg16.json'), 'w') as json_file:
            json_file.write(model_json)
        network.save_weights(os.path.join(MODEL_SAVE_PATH, 'vgg16.h5'))

    def load_local_resources(self):
        """
        Loads initial ImageNet trained network which is used
        as a base for own custom classifier.
        Returns ImageNet-trained VGG16 neural network architecture instance.
        """
        with open(os.path.join(MODEL_SAVE_PATH, 'vgg16.json'), 'r') as json_file:
            network = model_from_json(json_file.read())
        network.load_weights(os.path.join(MODEL_SAVE_PATH, 'vgg16.h5'))
        return network

    def set_up_model(self):
        network = self.load_local_resources()
        for layer in network.layers[:]:
            layer.trainable = False
        model = keras.Sequential()
        model.add(network)
        model.add(layers.Flatten())
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.classes, activation='softmax'))
        model.layers[0].trainable = False
        self.callbacks = []
        if self.enable_early_stopping:
            self.callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            )
        model.compile(
            loss=self.loss, 
            optimizer=self.optimizer, 
            metrics=["accuracy"],
        )
        return model


    def fit(self, X, y):
        X, y = shuffle(X, y)
        encoded_y = pd.get_dummies(y)
        self.labels = encoded_y.columns
        return self.model.fit(
            X, encoded_y.values,
            batch_size=self.batch_size, 
            epochs=self.epochs, 
            validation_split=0.1,
            callbacks=self.callbacks
        )
    
    def get_prob_matrix(self, X):
        return self.model.predict(X)

    def predict(self, X):
        prob_preds = self.get_prob_matrix(X)
        prob_preds_rounded = np.zeros(prob_preds.shape)
        prob_preds_rounded[np.arange(prob_preds.shape[0]), 
                           np.argmax(prob_preds, axis=1)] = 1
        preds_dummy_df = pd.DataFrame(prob_preds_rounded)#, columns=self.labels)
        return preds_dummy_df.idxmax(axis=1).values