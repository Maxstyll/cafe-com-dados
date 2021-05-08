import argparse

from src import main

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, help='Tipo de Treino.', required=False, choices=['TensorFlow', 'PyTorch'], default='True')
    parser.add_argument('--aula', type=str, help='Número da Aula.', required=False, default='True')
    args = parser.parse_args()

    start = main.Main()

    type = args.type
    aula = args.aula

    if "TensorFlow" in type:
        start.tensorflow(int(aula))
    elif "PyTorch" in type:
        start.pytorch(int(aula))


if __name__ == '__main__':
    run()