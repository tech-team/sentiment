import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data
import numpy as np

mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/")
image_size = 28
n_labels = 10
n_channels = 1

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, n_channels)).astype(np.float32)
    labels = (np.arange(n_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(mnist.train.images, mnist.train.labels)
valid_dataset, valid_labels = reformat(mnist.validation.images, mnist.validation.labels)
test_dataset, test_labels = reformat(mnist.test.images, mnist.test.labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def create_w_b(n_prev_layer, n_next_layer):
    w = tf.Variable(tf.random_normal([n_prev_layer, n_next_layer]))
    b = tf.Variable(tf.random_normal([n_next_layer]))
    return w, b


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def tf_accuracy(predictions, labels, tf_accuracy_name='accuracy'):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return acc, tf.scalar_summary(tf_accuracy_name, acc)


def main():
    start_learn_rate = 0.01
    reg = 0.001
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, n_channels),
                                          name='tf_train_dataset')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_labels), name='tf_train_labels')

        tf_valid_dataset = tf.constant(valid_dataset, name='tf_valid_dataset')
        tf_valid_labels = tf.constant(valid_labels, name='tf_valid_labels')

        tf_test_dataset = tf.constant(test_dataset, name='tf_test_dataset')
        tf_test_labels = tf.constant(test_labels, name='tf_test_labels')

        layer1_weights = tf.Variable(tf.truncated_normal([10, 10, n_channels, 6], stddev=0.1), name="conv_w")
        layer1_biases = tf.Variable(tf.zeros([6]), name="conv_b")
        layer2_weights = tf.Variable(tf.truncated_normal([150, n_labels], stddev=0.1), name="fc_w")
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]), name="fc_b")

        def model(data):
            with tf.name_scope("conv2x2"):
                conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='VALID')

            with tf.name_scope("relu_1"):
                hidden = tf.nn.relu(conv + layer1_biases)

            with tf.name_scope("max_pool2x2"):
                hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

            with tf.name_scope("reshape"):
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

            with tf.name_scope("fc"):
                hidden = tf.matmul(reshape, layer2_weights) + layer2_biases
            return hidden

        # Training computation.
        logits = model(tf_train_dataset)
        with tf.name_scope("xent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        with tf.name_scope("train_prediction"):
            train_prediction = tf.nn.softmax(logits)

        with tf.name_scope("valid_prediction"):
            valid_prediction = tf.nn.softmax(model(tf_valid_dataset))

        with tf.name_scope("test_prediction"):
            test_prediction = tf.nn.softmax(model(tf_test_dataset))

        batch_accuracy, batch_accuracy_summary = tf_accuracy(train_prediction, tf_train_labels, 'batch_accuracy')
        valid_accuracy, valid_accuracy_summary = tf_accuracy(valid_prediction, tf_valid_labels, 'valid_accuracy')

    num_steps = 3001

    with tf.Session(graph=graph) as session:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/tensorboard", session.graph_def)
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {
                tf_train_dataset: batch_data,
                tf_train_labels: batch_labels
            }
            _, l, predictions, batch_summary, batch_acc = session.run(
                [optimizer, loss, train_prediction, batch_accuracy_summary, batch_accuracy], feed_dict=feed_dict)

            writer.add_summary(batch_summary, step)
            if step % 500 == 0:
                valid_acc = valid_accuracy.eval()
                valid_acc_summary = valid_accuracy_summary.eval()
                print("Minibatch loss at step", step, ":", l)
                print("Minibatch accuracy: %.3f" % batch_acc)
                print("Validation accuracy: %.3f" % valid_acc)
                writer.add_summary(valid_acc_summary, step)
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


if __name__ == "__main__":
    main()
