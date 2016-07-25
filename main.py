import tensorflow as tf

from sentiment.cnn import SentimentCNN


def load_dataset(path):
    dataset = []
    labels = []
    with open(path, 'r') as f:
        texts = f.readlines()
    for text in texts:
        dataset.append(text)
        labels.append('+')
    return dataset, labels


def main():
    train_dataset, train_labels = load_dataset('data/tweets1.txt')
    with tf.Session() as session:
        cnn = SentimentCNN(
            session=session,
            embeddings_model_path='./sentiment/saved/model.ckpt-2264733',
            embeddings_vocab_path='./sentiment/saved/vocab.txt',
            embeddings_size=200,
            sentence_length=60,
            n_labels=2
        )

        cnn.train(train_dataset=train_dataset, train_labels=train_labels)

if __name__ == '__main__':
    main()
