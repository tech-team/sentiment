import os

import datetime
import tensorflow as tf
import numpy as np

from sentiment import SentimentAnalysisModel
from sentiment.w2v_model import Word2VecModel


class SentimentCNN(SentimentAnalysisModel):
    MODEL_FILE_NAME = 'sentiment_model.ckpt'

    WRITE_SUMMARY = False

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
                 dropout_keep_prob=0.5,
                 l2_lambda=0.0,
                 learning_rate=0.05,
                 batch_size=10,
                 n_steps=10000,
                 validation_check_steps=500,
                 summary_path='/tmp/tensorboard',
                 model_save_path=None):
        super().__init__()
        self.session = session
        self.sentence_length = sentence_length
        self.n_labels = n_labels
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.filter_stride = filter_stride
        self.dropout_keep_prob_value = dropout_keep_prob
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.check_steps = validation_check_steps
        self.summary_path = summary_path
        self.model_save_path = model_save_path

        self._word2vec = Word2VecModel(sess=session)
        self.load_embeddings(embeddings_model_path, embeddings_vocab_path, embeddings_size)
        self.embeddings_shape = self._word2vec.get_embeddings_shape()

        self._x = None
        self._y = None
        self._dropout_keep_prob = None
        self._logits = None
        self._loss = None
        self._prediction = None
        self._accuracy = None
        self._optimizer = None
        self._w = []
        self._b = []
        self.saver = None
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
                self._w.append(weights)
                self._b.append(bias)

                conv = tf.nn.conv2d(embed, weights, list(self.filter_stride), padding='VALID', name='conv')
                relu = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
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
            fc_weights = tf.Variable(tf.truncated_normal([h_shape[1], self.n_labels], stddev=0.1), name="w")
            fc_biases = tf.Variable(tf.constant(0.0, shape=[self.n_labels]), name="b")
            h = tf.matmul(h, fc_weights) + fc_biases

            self._w.append(fc_weights)
            self._b.append(fc_biases)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, self._dropout_keep_prob)
        return h

    def loss(self, logits, labels, weights=None, biases=None):
        with tf.name_scope("xent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

            l2_reg = 0.0
            if weights:
                l2_reg += sum(tf.map_fn(tf.nn.l2_loss, weights))
            if biases:
                l2_reg += sum(tf.map_fn(tf.nn.l2_loss, biases))
            loss += self.l2_lambda * l2_reg
        return loss

    def optimze(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def build_graph(self):
        self._x = tf.placeholder(tf.int32, shape=(None, self.sentence_length), name='x')
        self._y = tf.placeholder(tf.float32, shape=(None, self.n_labels), name='y')
        self._dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')

        self._logits = self.create_model(self._x)
        self._prediction = tf.nn.softmax(self._logits, name='prediction')
        self._accuracy = self.tf_accuracy(self._prediction, self._y)
        self._loss = self.loss(self._logits, self._y)
        self._optimizer = self.optimze(self._loss)

        if self.model_save_path is None:
            print('WARNING: model_save_path is not specified, model won\'t be saved!')
        else:
            self.saver = tf.train.Saver()

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
        if has_test_set:
            print('Test dataset: size = {}; shape = {}'.format(len(test_dataset), test_dataset.shape))

        tf_train_loss_summary = tf.scalar_summary("train_loss", self._loss)
        tf_valid_loss_summary = tf.scalar_summary("valid_loss", self._loss)
        tf_train_accuracy_summary = tf.scalar_summary('train_accuracy', self._accuracy)
        tf_valid_accuracy_summary = tf.scalar_summary('valid_accuracy', self._accuracy)

        if self.WRITE_SUMMARY:
            writer = tf.train.SummaryWriter(self.summary_path, self.session.graph)

        tf.initialize_all_variables().run(session=self.session)

        loss, accuracy = 0, 0
        for step in range(self.n_steps + 1):
            offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)
            batch_data = train_dataset[offset:(offset + self.batch_size), :]
            batch_labels = train_labels[offset:(offset + self.batch_size), :]

            feed_dict = {
                self._x: batch_data,
                self._y: batch_labels,
                self._dropout_keep_prob: self.dropout_keep_prob_value
            }
            _, loss, accuracy, loss_summary, accuracy_summary = self.session.run(
                [self._optimizer, self._loss, self._accuracy, tf_train_loss_summary, tf_train_accuracy_summary],
                feed_dict=feed_dict
            )

            if writer:
                writer.add_summary(loss_summary, step)
                writer.add_summary(accuracy_summary, step)
            print("{}: step {}, loss {:g}, accuracy {:g}".format(datetime.datetime.now().isoformat(),
                                                                 step, loss, accuracy))
            if step % self.check_steps == 0:
                if has_validation_set is not None:
                    feed_dict = {
                        self._x: valid_dataset,
                        self._y: valid_labels,
                        self._dropout_keep_prob: 1.0
                    }
                    loss, accuracy, loss_summary, accuracy_summary = self.session.run(
                        [self._loss, self._accuracy, tf_valid_loss_summary, tf_valid_accuracy_summary],
                        feed_dict=feed_dict
                    )

                    if writer:
                        writer.add_summary(loss_summary, step)
                        writer.add_summary(accuracy_summary, step)
                    print()
                    print("VALIDATION: {}: step {}, loss {:g}, accuracy {:g}".format(datetime.datetime.now().isoformat(),
                                                                                     step, loss, accuracy))
                    print()

        if self.model_save_path and self.saver:
            self.save()

        return loss, accuracy

    def save(self):
        if self.model_save_path and self.saver:
            save_path = self.saver.save(self.session, os.path.join(self.model_save_path, self.MODEL_FILE_NAME))
            print('Model saved in file: {}'.format(save_path))
            return save_path
        else:
            raise Exception('Can\'t save: model_save_path is None')

    def restore(self):
        if self.model_save_path and self.saver:
            full_path = os.path.join(self.model_save_path, self.MODEL_FILE_NAME)
            self.saver.restore(self.session, full_path)
            print('Model restored from file: {}'.format(full_path))
        else:
            raise Exception('Can\'t restore: model_save_path is None')

    def predict(self, words):
        words, _ = self.prepare_dataset(np.asarray([words]))
        feed_dict = {
            self._x: words,
            self._dropout_keep_prob: 1.0
        }
        prediction = self.session.run(
            [self._prediction],
            feed_dict=feed_dict
        )
        return prediction[0][0]

    @staticmethod
    def tf_accuracy(predictions, labels, tf_accuracy_name='accuracy'):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
            acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return acc

    @staticmethod
    def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

    def prepare_dataset(self, dataset, labels=None):
        if dataset is None and labels is None:
            return None, None

        assert labels is None or dataset.shape[0] == labels.shape[0]

        processed_dataset = np.ndarray((len(dataset), self.sentence_length), dtype=np.int32)
        if labels is not None:
            processed_labels = np.ndarray(labels.shape, dtype=np.int32)
        else:
            processed_labels = None

        real_dataset_length = 0
        empty_sents = 0
        for i, source_words in enumerate(dataset):
            words = self._word2vec.word2id_many(source_words)
            if words:
                if len(words) < self.sentence_length:
                    sentence_padding = self.sentence_length - len(words)
                    words = np.pad(words, (0, sentence_padding), mode='constant')
                elif len(words) > self.sentence_length:
                    words = words[:self.sentence_length]
                processed_dataset[real_dataset_length, :] = words
                if processed_labels is not None:
                    processed_labels[real_dataset_length] = labels[i]
                real_dataset_length += 1
            elif len(words) == 0:
                empty_sents += 1

        processed_dataset = processed_dataset[:real_dataset_length, :]
        if processed_labels is not None:
            processed_labels = processed_labels[:real_dataset_length]
            return processed_dataset, processed_labels[:real_dataset_length]
        return processed_dataset, None

    def close(self):
        self.session.close()

