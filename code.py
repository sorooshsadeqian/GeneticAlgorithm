import re
import string
import random
from collections import defaultdict
import numpy as np
from functools import reduce
import time


DEFAULT_POPULATION_SIZE = 200
ELEMENT_BY_ELEMENT = 1
SINGLE_POINT_CROSSOVER = 2
MULTIPOINT_CROSSOVER = 3


class Decoder:
    _encoded_text: str = None
    _population_size = None

    def __init__(self, text: str, population_size: int = DEFAULT_POPULATION_SIZE):
        self._population_size = population_size
        self._encoded_text = text

    def create_initial_population(self) -> list:
        return [self.create_random_chromosome() for i in range(self._population_size)]

    @staticmethod
    def get_alphabet_list():
        return list(string.ascii_lowercase)

    @classmethod
    def create_random_chromosome(cls):
        chromosome = []
        alphabet = cls.get_alphabet_list()
        alpha_count = len(alphabet)
        for i in range(alpha_count):
            chromosome.append(random.choice(alphabet))
            alphabet.remove(chromosome[-1])
        return tuple(chromosome)

    @staticmethod
    def build_clean_dictionary(file_path: str = './global_text.txt') -> dict:
        with open(file_path, 'r') as f:
            original_text = f.read()
        cleaned_text = re.split('\W+', original_text)
        dictionary = defaultdict(set)

        for word in cleaned_text:
            if len(word) > 1:
                dictionary[len(word)].add(word.lower())
        return dictionary

    @staticmethod
    def decode_word(word, chromosome):
        result = ''
        for i in word:
            if i.isalpha():
                result += chromosome[ord(i) - ord('a')]
            else:
                result += i
        return result

    @classmethod
    def calculate_fitness_score(cls, text, chromosome, dictionary):
        fitness_score = 0
        for word in text:
            decoded_word = cls.decode_word(word, chromosome)
            if decoded_word in dictionary[len(word)]:
                fitness_score += len(word)
        return fitness_score

    @classmethod
    def calculate_fitness_scores(cls, text, population, dictionary, already_calculated):
        scores = []
        for chromosome in population:
            if chromosome not in already_calculated:
                score = cls.calculate_fitness_score(text, chromosome, dictionary)
                scores.append(score)
            else:
                scores.append(already_calculated.get(chromosome))
        return scores

    @staticmethod
    def crossover(player_1, player_2):
        chromosome_length = len(player_1)
        crossover_point_1 = random.randint(0, chromosome_length - 2)
        crossover_point_2 = random.randint(crossover_point_1 + 1, chromosome_length - 1) + 1
        child_1 = [0 for _ in range(chromosome_length)]
        child_2 = [0 for _ in range(chromosome_length)]
        child_1[crossover_point_1:crossover_point_2] = player_1[crossover_point_1:crossover_point_2]
        child_2[crossover_point_1:crossover_point_2] = player_2[crossover_point_1:crossover_point_2]
        i = crossover_point_2 % chromosome_length
        j = i
        while j > crossover_point_2 - 1 or j < crossover_point_1:
            if player_2[i] not in child_1:
                child_1[j] = player_2[i]
                j = (j + 1) % chromosome_length
            i = (i + 1) % chromosome_length
        i = crossover_point_2 % chromosome_length
        j = i
        while j > crossover_point_2 - 1 or j < crossover_point_1:
            if player_1[i] not in child_2:
                child_2[j] = player_1[i]
                j = (j + 1) % chromosome_length
            i = (i + 1) % chromosome_length
        return [child_1, child_2]

    @staticmethod
    def mutate(_chromosome):
        mutation_distribution = [0] * 50 + [1] * 30 + [2] * 15 + [3] * 5
        chromosome = _chromosome
        genes_to_mutate = random.choice(mutation_distribution)
        already_mutated_genes = []
        for i in range(genes_to_mutate):
            first_gene = random.randint(0, len(chromosome) - 1)
            while first_gene in already_mutated_genes:
                first_gene = random.randint(0, len(chromosome) - 1)
            already_mutated_genes.append(first_gene)
            second_gene = random.randint(0, len(chromosome) - 1)
            while second_gene in already_mutated_genes:
                second_gene = random.randint(0, len(chromosome) - 1)
            already_mutated_genes.append(second_gene)
            chromosome[first_gene], chromosome[second_gene] = chromosome[second_gene], chromosome[first_gene]
        return chromosome
        

    @classmethod
    def generate_offspring(cls, player_1: list, player_2: list):
        children = cls.crossover(player_1, player_2)
        for i in range(len(children)):
            children[i] = tuple(cls.mutate(children[i]))
        return children

    def find_key(self, text, _population, reference):
        text = re.split('\W+', text.lower())
        text = list(filter(lambda word: len(word) > 1, text))
        population = _population
        threshold = sum(list(map(len, text))) * 99 // 100
        highest_fitness = 0
        already_calculated = {}
        generations = 0
        not_changed = 0
        while highest_fitness < threshold:
            if not_changed > 30:
                highest_fitness = 0
                not_changed = 0
                population = self.create_initial_population()
            generations += 1
            scores = self.calculate_fitness_scores(text, population, reference, already_calculated)
            for i in range(len(scores)):
                already_calculated[population[i]] = scores[i]
            sorted_indices = list(reversed(np.argsort(scores)))
            if scores[sorted_indices[0]] > highest_fitness:
                highest_fitness = scores[sorted_indices[0]]
                not_changed = 0
            else:
                not_changed += 1
            sorted_population = [population[i] for i in sorted_indices]
            new_generation = [*sorted_population[:len(sorted_population) // 10]]
            while len(new_generation) < len(sorted_population):
                first_chromosome = random.choice(
                    sorted_population[:len(sorted_population) // 5])
                second_chromosome = random.choice(
                    sorted_population[:len(sorted_population) // 5])
                children = self.generate_offspring(
                    first_chromosome, second_chromosome)
                for i in children:
                    if len(new_generation) < len(population):
                        new_generation.append(i)
            population = new_generation
        return sorted_population[0]

    @classmethod
    def create_key_dict(cls, key):
        alpha = cls.get_alphabet_list()
        key_dict = {}
        key_length = len(key)
        for i in range(key_length):
            key_dict[alpha[i]] = key[i]
            key_dict[alpha[i].upper()] = key[i].upper()
        return key_dict

    def decode_text(self, key):
        decoded_text = [key.get(char, char) for char in self._encoded_text]
        return ''.join(decoded_text)

    def decode(self):
        reference = self.build_clean_dictionary()
        text = self._encoded_text
        population = self.create_initial_population()
        start = time.time()
        key = self.find_key(text, population, reference)
        end = time.time()
        key_dict = self.create_key_dict(key)
        decoded_text = self.decode_text(key_dict)
        return decoded_text