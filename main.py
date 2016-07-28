import tensorflow as tf

from data_utils import load_dataset, text2words
from sentiment.cnn import SentimentCNN


def main():
    # save_dataset(from_path='data/training.1600000.processed.noemoticon.csv',
    #              to_path='data/preprocessed.json',
    #              size=None)
    #
    # return

    train_dataset, train_labels, valid_dataset, valid_labels = \
        load_dataset('data/preprocessed.json', preprocess=False, size=10000)

    max_len = max(map(len, train_dataset))
    print('Maximum sentence length: {}'.format(max_len))

    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as session:
            cnn = SentimentCNN(
                session=session,
                embeddings_model_path='./sentiment/saved/model.ckpt-2264733',
                embeddings_vocab_path='./sentiment/saved/vocab.txt',
                embeddings_size=200,
                sentence_length=max_len,
                n_labels=2,
                filter_sizes=(3, 4, 5),
                dropout_keep_prob=0.4,
                l2_lambda=0.1,
                n_filters=128,
                n_steps=2000,
                batch_size=128,
                learning_rate=0.003,
                validation_check_steps=100,
                summary_path='./summary'
            )

            cnn.train(train_dataset=train_dataset, train_labels=train_labels,
                      valid_dataset=valid_dataset, valid_labels=valid_labels)

            run_interactive(cnn)


def run_interactive(cnn):
    while True:
        try:
            text = input('Text: ')
            words = text2words(text)
            prediction = cnn.predict(words)
            print('Negative: {:g}. Positive: {:g}'.format(prediction[0], prediction[1]))
            print()
        except KeyboardInterrupt:
            return
        except:
            pass

if __name__ == '__main__':
    main()
