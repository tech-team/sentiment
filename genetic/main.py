from cnn_model import load_config
from genetic.genetic import Genetic


def main():
    base_config = load_config('meta_config.yaml')

    genetic = Genetic(10, 0.1, base_config)
    genetic.start()
