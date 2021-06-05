from src.tensorflow.customlayers import CustomLayers
from src.tensorflow.neuralnetwork import NeuralNetwork
from src.tensorflow.convnet import Convnet
from src.tensorflow.regularization import Regularization
from src.tensorflow.callbacks import Callbacks

class Main:
    switcherTensorflow = {
        1: NeuralNetwork(),
        2: Convnet(),
        3: Regularization(),
        4: CustomLayers(),
        5: Callbacks()
    }

    def __init__(self):
        pass
    
    def tensorflow(self, nunberAula):
        aula = self.switcherTensorflow.get(nunberAula, "Invalid month")
        aula.run()
