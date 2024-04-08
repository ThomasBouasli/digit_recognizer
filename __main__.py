from diy import DIYNeuralNetwork
from tf import TensorFlowNeuralNetwork

if __name__ == '__main__':
    diy_model = DIYNeuralNetwork.from_scratch()

    diy_model.train('data/train.csv', 500, 0.1)

    diy_model.test_model('data/train.csv')

    diy_model.save_model('models/diy/')

    tf_model = TensorFlowNeuralNetwork.from_scratch()

    tf_model.train('data/train.csv', 10, 32)

    tf_model.test_model('data/train.csv')

    tf_model.save_model('models/tf/model.h5')