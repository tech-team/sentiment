import tensorflow as tf
import numpy as np

from sentiment import SentimentAnalysisModel


class SentimentCNN(SentimentAnalysisModel):
    def __init__(self,
                 session,
                 embeddings_path,
                 sentence_length,
                 n_labels,
                 filter_sizes=(3,),
                 n_filters=1,
                 batch_size=10):
        super().__init__()
        self.session = session
        self.sentence_length = sentence_length
        self.n_labels = n_labels
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.batch_size = batch_size
        self.embeddings = self.load_embeddings(embeddings_path)
        self.embeddings_shape = self.embeddings.get_shape()

        self._train_data = None
        self._train_labels = None
        self._logits = None
        self._loss = None
        self._train = None
        self.saver = None

    def load_embeddings(self, path):
        return tf.constant(np.zeros((1000, 128)), name='embeddings')

    def create_model(self, data):
        train_set_shape = (self.batch_size, self.sentence_length, self.embeddings_shape[1], 1)
        train_set = tf.placeholder(tf.int32, shape=train_set_shape, name='train_set')
        train_labels = tf.placeholder(tf.int32, shape=(self.batch_size, self.n_labels), name='train_labels')
        return train_set # TODO

    def loss(self, logits, labels):
        with tf.name_scope("xent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        return loss

    def optimze(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        return optimizer

    def build_graph(self):
        self._train_data = [] # TODO
        self._train_labels = [] # TODO
        self._logits = self.create_model(self._train_data)
        self._loss = self.loss(self._logits, self._train_labels)
        self._train = self.optimze(self.loss)

        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()

    def train(self):
        pass
