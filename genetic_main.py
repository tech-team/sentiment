from cnn_model import load_config
from data_utils import load_rubtsova_datasets
from genetic.genetic import Genetic


def main():
    config = load_config('meta_config.yaml')

    cnn_base_config = config['cnn']
    ds_config = config['datasets']

    datasets = load_rubtsova_datasets(ds_config['positive'],
                                      ds_config['negative'],
                                      ds_config['size'])

    genetic = Genetic(
        dict(
            steps=2,
            mutation_prob=0.1,
            population_size=10,
            crossovers_count=6
        ),
        cnn_base_config=cnn_base_config,
        datasets=datasets
    )

    genetic.start()

if __name__ == '__main__':
    main()
