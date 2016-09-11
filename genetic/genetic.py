from genetic.chromosome import Chromosome


class Genetic:
    def __init__(self, population_size=10, mutation_prob=0.1, steps=10, base_config='meta_config.yaml'):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.steps = steps
        self.base_config = base_config

        self.population = []

    def start(self):
        self.create_population()
        for i in range(self.steps):
            self.step()

    def create_population(self):
        self.population = []
        for i in range(self.population_size):
            self.population.append(Chromosome.create_random(self.base_config))

    def step(self):
        pass
