import numpy as np
from utils.data_parser import DataParser


class ActivationFunction:
    @staticmethod
    def reLu(x):
        return np.maximum(0,x)
    
    @staticmethod
    def derivative_reLu(x):
        return x > 0

    @staticmethod
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))


class DIYNeuralNetwork:

    def __init__(self, weight_1, bias_1, weight_2, bias_2):
        self.__weight_1 = weight_1
        self.__bias_1 = bias_1
        self.__weight_2 = weight_2
        self.__bias_2 = bias_2

    @staticmethod
    def from_model(directory = ''):
        weight_1 = np.load(directory + 'weight_1.npy')
        bias_1 = np.load(directory + 'bias_1.npy')
        weight_2 = np.load(directory + 'weight_2.npy')
        bias_2 = np.load(directory + 'bias_2.npy')
        return DIYNeuralNetwork(weight_1, bias_1, weight_2, bias_2)
        

    @staticmethod
    def from_scratch():
        weight_1, bias_1, weight_2, bias_2 = DIYNeuralNetwork.__init_params()
        return DIYNeuralNetwork(weight_1, bias_1, weight_2, bias_2)

    def save_model(self, directory = ''):
        np.save(directory + 'weight_1', self.__weight_1)
        np.save(directory + 'bias_1', self.__bias_1)
        np.save(directory + 'weight_2', self.__weight_2)
        np.save(directory + 'bias_2', self.__bias_2)
    
    def train(self, train_data_file_name, iterations, alpha):
        x, y = DataParser.parse(train_data_file_name)
        self.__weight_1, self.__bias_1, self.__weight_2, self.__bias_2 = self.__gradient_descent(x, y, iterations, alpha)

    def test_model(self, test_data_file_name):
        x, y = DataParser.parse(test_data_file_name)
        predictions = self.predict(x)
        accuracy = self.__get_accuracy(predictions, y)
        print('Accuracy:', accuracy)
        

    def predict(self, x):
        _, _, _, a_2 = self.__forward_propagation(x, self.__weight_1, self.__bias_1, self.__weight_2, self.__bias_2)
        return self.__get_predictions(a_2)

    @staticmethod
    def __init_params():
        weight_1 = np.random.rand(10,784) - 0.5
        bias_1 = np.random.rand(10,1) - 0.5
        weight_2 = np.random.rand(10,10) - 0.5  
        bias_2 = np.random.rand(10,1) - 0.5
        return weight_1, bias_1, weight_2, bias_2

    def __gradient_descent(self, x, y, iterations, alpha):
        weight_1, bias_1, weight_2, bias_2 = self.__weight_1, self.__bias_1, self.__weight_2, self.__bias_2
        for i in range(iterations):
            a_1, z_1, a_2, _ = self.__forward_propagation(x, weight_1, bias_1, weight_2, bias_2)
            dw_1, db_1, dw_2, db_2 = self.__backward_propagation(z_1, a_1, a_2, weight_2, y, x)
            weight_1, bias_1, weight_2, bias_2 = self.__update_params(weight_1, bias_1, weight_2, bias_2, dw_1, db_1, dw_2, db_2, alpha)
            if i % 10 == 0:
                _, _, a_2, _ = self.__forward_propagation(x, weight_1, bias_1, weight_2, bias_2)
                print('Iteration:', i)
                print('Accuracy:', self.__get_accuracy(self.__get_predictions(a_2), y))
        return weight_1, bias_1, weight_2, bias_2

    def __forward_propagation(self, x, weight_1, bias_1, weight_2, bias_2):
        z_1 = weight_1.dot(x) + bias_1
        a_1 = ActivationFunction.reLu(z_1) 
        z_2 = weight_2.dot(a_1) + bias_2
        a_2 = ActivationFunction.softmax(z_2)
        return a_1, z_1 ,a_2, z_2

    def __backward_propagation(self, z_1, a_1, a_2, weight_2, y, x):
        m = y.size
        one_hot_y = self.__one_hot(y)
        dz_2 = a_2 - one_hot_y
        dw_2 = 1/m * dz_2.dot(a_1.T)
        db_2 = 1/m * np.sum(dz_2)
        dz_1 = weight_2.T.dot(dz_2) * ActivationFunction.derivative_reLu(z_1)
        dw_1 = 1/m * dz_1.dot(x.T)
        db_1 = 1/m * np.sum(dz_1)
        return dw_1, db_1, dw_2, db_2

    def __one_hot(self, y):
        one_hot_y = np.zeros((y.size, y.max()+1))
        one_hot_y[np.arange(y.size),y] = 1
        one_hot_y = one_hot_y.T
        return one_hot_y

    def __update_params(self, weight_1, bias_1, weight_2, bias_2, dw_1, db_1, dw_2, db_2, alpha):
        weight_1 = weight_1 - alpha * dw_1
        bias_1 = bias_1 - alpha * db_1
        weight_2 = weight_2 - alpha * dw_2
        bias_2 = bias_2 - alpha * db_2
        return weight_1, bias_1, weight_2, bias_2

    def __get_predictions(self, a_2):
        return np.argmax(a_2, axis=0)
    
    def __get_accuracy(self, predictions, y):
        return np.mean(predictions == y)
    
