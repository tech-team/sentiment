import re

import tensorflow as tf
import numpy as np
import time

from sentiment import SentimentAnalysisModel
from sentiment.w2v_model import Word2VecModel


class SentimentCNN(SentimentAnalysisModel):
    def __init__(self,
                 session,
                 embeddings_model_path,
                 embeddings_vocab_path,
                 embeddings_size,
                 sentence_length,
                 n_labels,
                 filter_sizes=(3,),
                 n_filters=1,
                 filter_stride=(1, 1, 1, 1),
                 learning_rate=0.05,
                 batch_size=10,
                 n_steps=10000,
                 validation_check_steps=500,
                 summary_path='/tmp/tensorboard'):
        super().__init__()
        self.session = session
        self.sentence_length = sentence_length
        self.n_labels = n_labels
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.filter_stride = filter_stride
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.check_steps = validation_check_steps
        self.summary_path = summary_path

        self._word2vec = Word2VecModel(sess=session)
        self.load_embeddings(embeddings_model_path, embeddings_vocab_path, embeddings_size)
        self.embeddings_shape = self._word2vec.get_embeddings_shape()

        self._train_dataset = None
        self._train_labels = None
        self._logits = None
        self._loss = None
        self._train = None
        # self.saver = None
        self.build_graph()

    def load_embeddings(self, model_path, vocab_path, embeddings_size):
        self._word2vec.load_model(model_path, vocab_path, embeddings_size)

    def create_model(self, data):
        embed = tf.nn.embedding_lookup(self._word2vec.w_in, data, name='embedding')
        embed = tf.expand_dims(embed, -1)

        filters = []
        for filter_id, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-{}-{}'.format(filter_id, filter_size)):
                weights = tf.Variable(
                    tf.truncated_normal([filter_size, self.embeddings_shape[1], 1, self.n_filters], stddev=0.1),
                    name='w'
                )
                bias = tf.Variable(tf.zeros([self.n_filters]), name='b')

                conv = tf.nn.conv2d(embed, weights, list(self.filter_stride), padding='VALID', name='conv')
                relu = tf.nn.relu(conv + bias, name='relu')
                pool = tf.nn.max_pool(relu,
                                      [1, self.sentence_length - filter_size + 1, 1, 1],
                                      [1, 1, 1, 1],
                                      padding='VALID',
                                      name='pool')
                filters.append(pool)

        concat = tf.concat(3, filters)
        h = tf.reshape(concat, [-1, len(filters) * self.n_filters])
        h_shape = h.get_shape().as_list()

        with tf.name_scope("fc"):
            fc_weights = tf.Variable(tf.truncated_normal([h_shape[1], self.n_labels], stddev=0.1), name="fw")
            fc_biases = tf.Variable(tf.constant(0.0, shape=[self.n_labels]), name="b")
            h = tf.matmul(h, fc_weights) + fc_biases

        return h

    def loss(self, logits, labels):
        with tf.name_scope("xent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        return loss

    def optimze(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def build_graph(self):
        train_set_shape = (self.batch_size, self.sentence_length)
        self._train_dataset = tf.placeholder(tf.int32, shape=train_set_shape, name='train_data')
        self._train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_labels), name='train_labels')

        self._logits = self.create_model(self._train_dataset)
        self._loss = self.loss(self._logits, self._train_labels)
        self._train = self.optimze(self._loss)

        # self.saver = tf.train.Saver()

    def train(self,
              train_dataset, train_labels,
              valid_dataset=None, valid_labels=None,
              test_dataset=None, test_labels=None):
        train_dataset, train_labels = self.prepare_dataset(train_dataset, train_labels)
        valid_dataset, valid_labels = self.prepare_dataset(valid_dataset, valid_labels)
        test_dataset, test_labels = self.prepare_dataset(test_dataset, test_labels)

        has_validation_set = valid_dataset is not None and valid_labels is not None
        has_test_set = test_dataset is not None and test_labels is not None

        print('Train dataset: size = {}; shape = {}'.format(len(train_dataset), train_dataset.shape))
        if has_validation_set:
            print('Valid dataset: size = {}; shape = {}'.format(len(valid_dataset), valid_dataset.shape))
            valid_dataset = tf.constant(valid_dataset, name='valid_dataset')
            valid_labels = tf.constant(valid_labels, name='valid_labels')
        if has_test_set:
            print('Test dataset: size = {}; shape = {}'.format(len(test_dataset), test_dataset.shape))
            test_dataset = tf.constant(test_dataset, name='test_dataset')
            test_labels = tf.constant(test_labels, name='test_labels')

        train_prediction = tf.nn.softmax(self._logits, name='train_prediction')
        batch_accuracy, batch_accuracy_summary = self.tf_accuracy(train_prediction, self._train_labels,
                                                                  'batch_accuracy')

        if has_validation_set:
            valid_prediction = tf.nn.softmax(self.create_model(valid_dataset), name='valid_prediction')
            valid_accuracy, valid_accuracy_summary = self.tf_accuracy(valid_prediction, valid_labels, 'valid_accuracy')
        else:
            valid_accuracy, valid_accuracy_summary = None, None
        if has_test_set:
            test_prediction = tf.nn.softmax(self.create_model(test_dataset), name='test_prediction')
        else:
            test_prediction = None

        writer = tf.train.SummaryWriter(self.summary_path, self.session.graph_def)

        tf.initialize_all_variables().run()
        for step in range(self.n_steps):
            offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)
            batch_data = train_dataset[offset:(offset + self.batch_size), :]
            batch_labels = train_labels[offset:(offset + self.batch_size), :]

            feed_dict = {
                self._train_dataset: batch_data,
                self._train_labels: batch_labels
            }
            _, loss, predictions, batch_summary, batch_acc = self.session.run(
                [self._train, self._loss, train_prediction, batch_accuracy_summary, batch_accuracy],
                feed_dict=feed_dict
            )

            writer.add_summary(batch_summary, step)
            if step % self.check_steps == 0:
                print("Minibatch loss at step", step, ":", loss)
                print("Minibatch accuracy: %.3f" % batch_acc)
                if valid_accuracy is not None:
                    valid_acc = valid_accuracy.eval()
                    valid_acc_summary = valid_accuracy_summary.eval()
                    print("Validation accuracy: %.3f" % valid_acc)
                    writer.add_summary(valid_acc_summary, step)
                print()
        if test_prediction is not None:
            print("Test accuracy: %.1f%%" % self.accuracy(test_prediction.eval(), test_labels))

    @staticmethod
    def tf_accuracy(predictions, labels, tf_accuracy_name='accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return acc, tf.scalar_summary(tf_accuracy_name, acc)

    @staticmethod
    def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

    def prepare_dataset(self, dataset, labels):
        if dataset is None and labels is None:
            return None, None

        assert dataset.shape[0] == labels.shape[0]

        processed_dataset = np.ndarray((len(dataset), self.sentence_length), dtype=np.int32)

        real_dataset_length = 0
        for i, words in enumerate(dataset):
            words = self._word2vec.word2id_many(words)
            if words is not None:
                sentence_padding = self.sentence_length - len(words)
                words = np.pad(words, (0, sentence_padding), mode='constant')
                processed_dataset[real_dataset_length, :] = words
                real_dataset_length += 1
            # if i % 100 == 0:
            #     print('Processed sentence {}/{}.'.format(
            #         i + 1,
            #         len(dataset)))

        processed_dataset = processed_dataset[:real_dataset_length, :]
        return processed_dataset, labels
