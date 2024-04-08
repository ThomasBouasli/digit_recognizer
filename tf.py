import tensorflow as tf
import numpy as np
from utils.data_parser import DataParser


class TensorFlowNeuralNetwork:
    def __init__(self, model = None):
        self.__model = model

    @staticmethod
    def from_scratch():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return TensorFlowNeuralNetwork(model)

    
    @staticmethod
    def from_model(file_name):
        model = tf.keras.models.load_model(file_name)
        return TensorFlowNeuralNetwork(model)

    def save_model(self, file_name):
        if self.__model is None:
            raise Exception('Model is not initialized')
        
        self.__model.save(file_name)

    def train(self, train_file_name, epochs, batch_size):
        if self.__model is None:
            raise Exception('Model is not initialized')

        x_train, y_train = DataParser.parse(train_file_name)

        self.__model.fit(x_train.T, y_train, epochs=epochs, batch_size=batch_size)

    def test_model(self, test_file_name):
        if self.__model is None:
            raise Exception('Model is not initialized')

        x, y = DataParser.parse(test_file_name)
        predictions = self.predict(x)
        accuracy = sum(predictions == y) / len(y)
        print('Accuracy:', accuracy)

    def predict(self, x):
        if self.__model is None:
            raise Exception('Model is not initialized')

        return np.argmax(self.__model.predict(x.T), axis=1)
