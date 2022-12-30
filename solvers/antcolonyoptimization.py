"""The ant colony optimization algorithm to solve the TSP problem"""
from typing import Iterator
from random import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from graph import Graph
from tsptypes import ShortestPath, AlgorithmResult


def ant_colony(graph: Graph, distances: dict[tuple[int, int], int],
               max_swarms: int, max_no_improvement: int,) -> Iterator[AlgorithmResult]:
    """Solve the TSP problem using ant colony optimization"""
    swarms_evaluated = 0
    evaluations_until_solved = 0
    swarms_without_improvement = 0
    node_count = len(graph.nodes)
    pheromones = initialize_pheromones(node_count, distances)
    for _ in range(max_swarms):
        if swarms_without_improvement >= max_no_improvement:
            break

        vertices, cycle_lengths = swarm_traversal(pheromones, distances)
        pheromones = pheromone_evaporation(pheromones)
        best_cycle_index = np.argmin(cycle_lengths)
        best_cycle = vertices[best_cycle_index]
        best_cycle_length = cycle_lengths[best_cycle_index]
        pheromones = pheromone_release(vertices, best_cycle, best_cycle_length, pheromones)
        pheromones = normalize(pheromones)
        swarms_evaluated += 1

        if cycle_lengths[best_cycle_index] < graph.optimal_cycle_length:
            graph.remove_vertices()
            graph.add_vertices(best_cycle)
            graph.optimal_cycle = ShortestPath(best_cycle_length, best_cycle)
            evaluations_until_solved = swarms_evaluated
            swarms_without_improvement = 0
            yield AlgorithmResult(swarms_evaluated, evaluations_until_solved)
        else:
            swarms_without_improvement += 1
    yield AlgorithmResult(swarms_evaluated, evaluations_until_solved)


def initialize_pheromones(node_count: int, distances: dict[tuple[int, int], int]) -> np.ndarray:
    """Initialize the pheromone array"""
    pheromones = np.zeros((node_count, node_count), dtype=float)
    for i, j in np.ndindex(pheromones.shape):
        if i != j:
            pheromones[i, j] = distances[(i, j)]
    for i in range(node_count):
        row_sum = sum(pheromones[i,:])
        for j in range(node_count):
            if i != j:
                pheromones[i, j] = row_sum / pheromones[i, j]
        row_sum = sum(pheromones[i,:])
        for j in range(node_count):
            if i != j:
                pheromones[i, j] /= row_sum
    return pheromones

def normalize(pheromones: np.ndarray) -> np.ndarray:
    """Normalize the pheromone matrix into probabilities"""
    node_count = pheromones.shape[0]
    for i in range(node_count):
        row_sum = sum(pheromones[i,:])
        for j in range(node_count):
            if i != j:
                pheromones[i, j] /= row_sum
    return pheromones


def swarm_traversal(pheromones: np.ndarray, distances: dict[tuple[int, int], int]
             ) -> tuple[np.ndarray, list[int]]:
    """Traverse the graph with a number of ants equal to the number of nodes"""
    node_count = pheromones.shape[0]
    swarm_size = node_count * node_count
    vertices = np.array([[(0, 0, 0)] * node_count] * swarm_size, dtype="i,i,i")
    cycle_lengths: list[int] = [0] * swarm_size
    # Traverse the graph swarm_size times
    with ThreadPoolExecutor(max_workers=min(swarm_size, 50)) as executor:
        futures = [executor.submit(traverse, node_count, pheromones, distances)
                    for i in range(swarm_size)]
        for i, completed in enumerate(as_completed(futures)):
            cycle_length, cycle = completed.result()
            vertices[i] = cycle
            cycle_lengths[i] = cycle_length
    return vertices, cycle_lengths


def traverse(node_count: int, pheromones: np.ndarray,
             distances: dict[tuple[int, int], int]) -> tuple[int, np.ndarray]:
    """Perform a traversal through the graph"""
    # Each traversal consists of node_count vertices
    current_node = 0
    visited = set([current_node])
    cycle_length = 0
    vertices = np.array([(0, 0, 0)] * node_count, dtype="i,i,i")
    for j in range(node_count-1):
        row_sorted_indices = np.argsort(pheromones[current_node,:])
        row = np.take(pheromones[current_node,:], row_sorted_indices)
        cumul = 0
        for k, _ in enumerate(row):
            if row[k] == 0.0:
                continue
            row[k], cumul = row[k] + cumul, row[k] + cumul

        index = -1
        chance = random()
        for k in range(1, len(row)):
            candidate = row_sorted_indices[k]
            if (row[k-1] < chance <= row[k]
                and candidate != current_node
                and candidate not in visited):
                index = candidate
                break
        # If no suitable index was found, the generated chance was probably too low
        # Pick the first index that's not itself and not visited yet
        if index == -1:
            for k in range(len(row)-1, 0, -1):
                candidate = row_sorted_indices[k]
                if candidate != current_node and candidate not in visited:
                    index = candidate
                    break

        distance = distances[current_node, index]
        vertices[j] = (current_node, index, distance)
        cycle_length += distance
        visited.add(index)
        current_node = index
    # Add the last vertex back to the starting node
    distance = distances[current_node, 0]
    vertices[node_count-1] = (current_node, 0, distance)
    cycle_length += distance
    return cycle_length, vertices


def pheromone_evaporation(pheromones: np.ndarray) -> np.ndarray:
    """Evaporation of pheromones after each traversal"""
    for index in np.ndindex(pheromones.shape):
        pheromones[index] *= 1 - pheromones[index]
    return pheromones


def pheromone_release(vertices: np.ndarray, best_cycle: np.ndarray,
                      best_cycle_length: int, pheromones: np.ndarray) -> np.ndarray:
    """Perform pheromone release, with elitism towards shorter cycles"""
    for cycle in vertices:
        for i, j, weight in cycle:
            pheromones[i, j] += 1 / weight
    for i, j, _ in best_cycle:
        pheromones[i, j] += 1 / best_cycle_length

    return pheromones
