import random

from genetic.chromosome import Chromosome


class Genetic:
    def __init__(self, genetic_config, cnn_base_config, datasets):
        self.genetic_config = genetic_config
        self.cnn_base_config = cnn_base_config
        self.datasets = datasets

        self.population = []
        self.step_id = None

    def start(self):
        self.step_id = 0
        self.create_population()
        for self.step_id in range(self.genetic_config['steps']):
            self.step()

        return self.population[0]

    def create_population(self):
        self.population = []
        for i in range(self.genetic_config['population_size']):
            self.population.append(Chromosome.create_random(self.cnn_base_config))

    def step(self):
        for ch in self.population:
            if ch.fitness is None:
                ch.evaluate(self.datasets)

        self.population.sort(key=lambda ch: -ch.fitness)

        self.print_best_config()

        if self._should_stop():
            # do not modify self.population in the end
            return

        children = []

        for i in range(self.genetic_config['crossovers_count']):
            mother, father = None, None
            while mother == father:
                mother = self._random_exp_parent()
                father = self._random_exp_parent()

            child = Chromosome.crossover(mother, father)
            if random.random() < self.genetic_config['mutation_prob']:
                child.mutate()

            children.append(child)

        best_parents_limit = self.genetic_config['population_size'] - len(children)
        self.population = children + self.population[:best_parents_limit]

    def print_best_config(self):
        print('-----Step {}-----'.format(self.step_id))
        ch = self.population[0]
        pprint(ch.config)
        print('Loss: {}'.format(ch.loss))
        print('Accuracy: {}'.format(ch.accuracy))
        print('-----------------')

    def _random_exp_parent(self):
        parent_id = random.expovariate(lambd=1.0)
        parent_id *= len(self.population)
        parent_id = int(parent_id)
        parent = self.population[parent_id]
        return parent

    def _should_stop(self):
        return self.step_id >= self.genetic_config['steps']
