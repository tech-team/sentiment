import tensorflow as tf

from cnn_model import load_config
from data_utils import load_rubtsova_datasets
from sentiment.cnn import SentimentCNN


def main():
    config = load_config('meta_config.yaml')

    cnn_base_config = config['cnn']
    ds_config = config['datasets']

    datasets = load_rubtsova_datasets(ds_config['positive'],
                                      ds_config['negative'],
                                      ds_config['size'])

    AVG_RUNS_COUNT = 10

    for n_filters in [20, 50, 100, 150, 200]:
        cnn_config = cnn_base_config.copy()
        cnn_config['n_filters'] = n_filters

        avg_accuracy = 0

        for i in range(AVG_RUNS_COUNT):
            loss, accuracy = evaluate(cnn_config, datasets)
            avg_accuracy += accuracy
            print('n_filters: {}, accuracy: {}'.format(n_filters, accuracy))

        avg_accuracy /= AVG_RUNS_COUNT

        print('n_filters: {}, avg_accuracy: {}'.format(n_filters, avg_accuracy))


def evaluate(config, datasets):
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as session:
            cnn = SentimentCNN(
                session=session,
                **config
            )

            loss, accuracy = cnn.train(*datasets)
            return loss, accuracy

if __name__ == '__main__':
    main()
