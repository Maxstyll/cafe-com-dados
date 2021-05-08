

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

    switcherPytorch = {
        1: "Invalid month"
    }

    def __init__(self):
        pass
    
    def tensorflow(self, nunberAula):
        aula = self.switcherTensorflow.get(nunberAula, "Invalid month")
        aula.run()

    def pytorch(self, nunberAula):
        aula = self.switcherPytorch.get(nunberAula, "Invalid month")
        print(aula)