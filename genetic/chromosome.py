import copy
import random
import tensorflow as tf
from sentiment.cnn import SentimentCNN


class Chromosome:
    def __init__(self, config, base_config):
        self.config = config  # genome
        self.base_config = base_config  # const ref
        self.fitness = None

        # just for output:
        self.loss = None
        self.accuracy = None

    @staticmethod
    def create_random(base_config):
        config = copy.deepcopy(base_config)

        # modify copy of base config
        config['filter_sizes'] = [
            random.randint(2, 6),
            random.randint(2, 6),
            random.randint(2, 6),
        ]
        config['dropout_keep_prob'] = random.random()
        config['l2_lambda'] = random.random()
        config['n_filters'] = random.randint(10, 500)
        config['batch_size'] = random.randint(10, 500)
        config['learning_rate'] = random.random()

        ch = Chromosome(config, base_config)
        return ch

    def clone(self):
        config = copy.deepcopy(self.config)
        return Chromosome(config, self.base_config)

    def mutate(self):
        keys_to_modify = random.sample(self.config.keys(), 3)
        random_source = Chromosome.create_random(self.base_config)

        for key in keys_to_modify:
            # it is a bit rude
            # we are completely replacing some genes by random ones
            # could be better to average between old and new value
            # but I'm too lazy for that
            # (note, that some params are ints, some - floats and some are even lists)
            self.config = random_source.config[key]

    @staticmethod
    def crossover(ch1, ch2):
        parents = [ch1, ch2]
        new_config = {}

        for key in ch1.config.keys():
            parent_id = random.randint(0, 1)
            new_config[key] = parents[parent_id].config[key]

        ch = Chromosome(new_config, ch1.base_config)
        return ch

    def evaluate(self, datasets):
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as session:
                cnn = SentimentCNN(
                    session=session,
                    **self.config
                )

                loss, accuracy = cnn.train(*datasets)
                self.update_fitness(loss, accuracy)

    def update_fitness(self, loss, accuracy):
        self.fitness = accuracy
        self.loss = loss
        self.accuracy = accuracy
        return self.fitness
