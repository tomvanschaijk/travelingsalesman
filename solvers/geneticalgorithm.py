"""The genetic algorithm to solve the TSP problem"""
from typing import Iterator
from sys import maxsize
from math import factorial
from random import shuffle, randint
import heapq
from graph import Graph


from tsptypes import ShortestPath, AlgorithmResult


def genetic_algorithm(graph: Graph, distances: dict[tuple[int, int], int],
                      population_size: int, max_generations: int, max_no_improvement: int,
                      ) -> Iterator[AlgorithmResult]:
    """Solve the TSP problem with a genetic algorithm"""
    generations_evaluated = 0
    generations_until_solved = 0
    generations_without_improvement = 0
    optimal_cycle: list[int] = []
    optimal_cycle_length = maxsize
    population = spawn(graph, population_size)
    cycle_lengths = get_cycle_lengths(population, distances)
    for _ in range(max_generations):
        if generations_without_improvement >= max_no_improvement:
            break
        fitness = determine_fitness(cycle_lengths)
        improved = False
        for i, cycle_length in enumerate(cycle_lengths):
            if cycle_length < optimal_cycle_length:
                improved = True
                optimal_cycle_length = cycle_length
                optimal_cycle = population[i]
        generations_evaluated += 1
        if improved:
            generations_until_solved = generations_evaluated
            generations_without_improvement = 0
            vertices: list[tuple[int, int, int]] = []
            node = optimal_cycle[0]
            for key in optimal_cycle[1:]:
                distance = distances[(node, key)]
                vertices.append((node, key, distance))
                node = key
            graph.remove_vertices()
            graph.add_vertices(vertices)
            graph.optimal_cycle = ShortestPath(optimal_cycle_length, vertices)
        else:
            generations_without_improvement += 1

        if len(population) == 1:
            break
        population, cycle_lengths = create_next_population(population, cycle_lengths,
                                                           fitness, distances)
        yield AlgorithmResult(generations_evaluated, generations_until_solved)
    yield AlgorithmResult(generations_evaluated, generations_until_solved)


def spawn(graph: Graph, population_size: int) -> list[list[int]]:
    """Create the initial generation"""
    start = graph[0].key
    keys = list(graph.nodes.keys())[1:]
    max_size = int(factorial(len(keys)) / 2)
    unique_permutations = set()
    while len(unique_permutations) < population_size and len(unique_permutations) < max_size:
        permutation = list(keys)
        shuffle(permutation)
        permutation = (start,) + tuple(permutation) + (start,)
        if permutation[::-1] not in unique_permutations:
            unique_permutations.add(permutation)
    return [list(permutation) for permutation in unique_permutations]


def create_next_population(current_population: list[list[int]], cycle_lengths: list[int],
                           fitness: list[float], distances: dict[tuple[int, int], int]
                           ) -> tuple[list[list[int]], list[int]]:
    """Create the next generation"""
    new_population: list[list[int]] = []
    population_size = len(current_population)

    # Create the offspring of the current generation
    offspring = create_offspring(current_population, fitness, population_size)

    # Perform a variation of elitism where we add the offspring to the current generation
    # and only continue with the fittest list of size population_size
    new_population = current_population + offspring
    offspring_cycle_lengths = get_cycle_lengths(offspring, distances)
    new_population_cycle_lengths = cycle_lengths + offspring_cycle_lengths
    new_population_fitness = fitness + determine_fitness(offspring_cycle_lengths)
    survivor_candidates = zip(new_population_fitness, new_population)
    fittest_indices = [i for _, i in heapq.nlargest(population_size, ((x, i)
                            for i, x in enumerate(survivor_candidates)))]
    new_population = [new_population[i] for i in fittest_indices]
    new_population_cycle_lengths = [new_population_cycle_lengths[i] for i in fittest_indices]
    return new_population, new_population_cycle_lengths


def create_offspring(current_population: list[list[int]], fitness: list[float],
                     population_size: int) -> list[list[int]]:
    """Create a new generation"""
    offspring: list[list[int]] = []
    while len(offspring) < population_size:
        parent1 = parent2 = 0
        while parent1 == parent2:
            parent1 = get_parent(fitness)
            parent2 = get_parent(fitness)

        child1 = crossover(current_population[parent1], current_population[parent2])
        child2 = crossover(current_population[parent2], current_population[parent1])

        child1 = mutate(child1)
        child2 = mutate(child2)

        offspring.append(child1)
        offspring.append(child2)
    return offspring


def get_parent(fitness: list[float]):
    """Get a parent using either tournament selection or biased random selection"""
    if randint(0, 1):
        return tournament_selection(fitness)
    return biased_random_selection(fitness)


def tournament_selection(fitness: list[float]) -> int:
    """Perform basic tournament selection to get a parent"""
    start, end = 0, len(fitness)-1
    candidate1 = randint(start, end)
    candidate2 = randint(start, end)
    while candidate1 == candidate2:
        candidate2 = randint(start, end)
    return candidate1 if fitness[candidate1] > fitness[candidate2] else candidate2


def biased_random_selection(fitness: list[float]) -> int:
    """Perform biased random selection to get a parent"""
    random_specimen = randint(0, len(fitness)-1)
    for i, _ in enumerate(fitness):
        if fitness[i] >= fitness[random_specimen]:
            return i
    return random_specimen


def crossover(parent1: list[int], parent2: list[int]) -> list[int]:
    """Cross-breed a new set of children from the given parents"""
    start = parent1[0]
    end = parent1[len(parent1)-1]
    parent1 = parent1[1:len(parent1)-1]
    parent2 = parent2[1:len(parent2)-1]
    split = randint(1, len(parent1)-1)
    child: list[int] = [0] * len(parent1)
    for i in range(split):
        child[i] = parent1[i]

    remainder = [i for i in parent2 if i not in child]
    for i, data in enumerate(remainder):
        child[split+i] = data
    return [start, *child, end]


def mutate(child: list[int]) -> list[int]:
    """Mutate the child sequence"""
    if randint(0, 1):
        child = swap_mutate(child)
    child = rotate_mutate(child)
    return child


def swap_mutate(child: list[int]) -> list[int]:
    """Mutate the cycle by swapping 2 nodes"""
    index1 = randint(1, len(child)-2)
    index2 = randint(1, len(child)-2)
    child[index1], child[index2] = child[index2], child[index1]
    return child


def rotate_mutate(child: list[int]) -> list[int]:
    """Mutate the cycle by rotating a part nodes"""
    split = randint(1, len(child)-2)
    head = child[0:split]
    mid = child[split:len(child)-1][::-1]
    tail = child[len(child)-1:]
    child = head+mid+tail
    return child


def get_cycle_lengths(population: list[list[int]],
                      distances: dict[tuple[int, int], int]) -> list[int]:
    """Get the lengths of all cycles in the graph"""
    cycle_lengths: list[int] = []
    for specimen in population:
        node = specimen[0]
        cycle_length = 0
        for key in specimen[1:]:
            key = int(key)
            cycle_length += distances[(node, key)]
            node = key
        cycle_lengths.append(cycle_length)
    return cycle_lengths


def determine_fitness(cycle_lengths: list[int]) -> list[float]:
    """Determine the fitness of the specimens in the population"""
    # Invert so that shorter paths get higher values
    fitness_sum = sum(cycle_lengths)
    fitness = [fitness_sum / cycle_length for cycle_length in cycle_lengths]

    # Normalize the fitness
    fitness_sum = sum(fitness)
    fitness = [f  / fitness_sum for f in fitness]

    return fitness
