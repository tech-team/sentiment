import tensorflow as tf
import tensorflow.python.framework.errors


class Word2VecModel:
    def __init__(self, sess):
        self.sess = sess
        self.w_in = None
        self.word2id = {}

    def load_model(self, model_file, vocab_file, emb_size):
        print('Loading word2vec vocab...')

        self.word2id = {}
        with open(vocab_file, 'r') as vocab_file:
            for i, line in enumerate(vocab_file):
                word, _ = line.split(' ')
                self.word2id[word] = i

        vocab_size = len(self.word2id)
        self.w_in = tf.Variable(tf.zeros([vocab_size, emb_size]), dtype=tf.float32, name="w_in")

        print('Loading word2vec model...')
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)

    def get_embeddings_constant(self):
        w_in_value = self.sess.run(self.w_in)
        return tf.constant(w_in_value, name='embeddings')

    def word2vec(self, word, run=True):
        if word in self.word2id:
            id = self.word2id[word]
            ids = tf.constant(id, dtype=tf.int32)
            vec = tf.nn.embedding_lookup(self.w_in, ids)

            if run:
                vec = self.sess.run(vec)

            return vec
        else:
            return None


def main():
    sess = tf.Session()
    model = Word2VecModel(sess)
    model.load_model(model_file='./saved/model.ckpt-2264733',
                     vocab_file='./saved/vocab.txt',
                     emb_size=200)

    model.w_in = model.get_embeddings_constant()

    print('Evaluating...')
    igor_vec = model.word2vec('igor')
    two_vec = model.word2vec('two')
    three_vec = model.word2vec('three')

    igor_dst = sess.run(tf.reduce_mean(tf.square(igor_vec - two_vec)))
    num_dst = sess.run(tf.reduce_mean(tf.square(three_vec - two_vec)))

    print('igor_dst = \'igor\' - \'two\' = {}'.format(igor_dst))
    print('num_dst = \'three\' - \'two\' = {}'.format(num_dst))


if __name__ == "__main__":
    main()
