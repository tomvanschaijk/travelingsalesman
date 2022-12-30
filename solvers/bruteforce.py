"""The brute force algorithm to solve the TSP problem"""
from typing import Iterator
from itertools import permutations
from graph import Graph
from tsptypes import ShortestPath, AlgorithmResult


def brute_force(graph: Graph, distances: dict[tuple[int, int], int]) -> Iterator[AlgorithmResult]:
    """Solve the TSP problem with a brute force implementation, running through all permutations"""
    unique_permutations = set()
    paths_evaluated = 0
    evaluations_until_solved = 0
    start = graph[0].key
    keys = list(graph.nodes.keys())
    sub_keys = keys[1:]
    for permutation in permutations(sub_keys):
        permutation = (start,) + permutation + (start,)
        if permutation not in unique_permutations and permutation[::-1] not in unique_permutations:
            paths_evaluated += 1
            unique_permutations.add(permutation)
            current_path_length = 0
            node = start
            vertices: list[tuple[int, int, int]] = []
            for key in permutation[1:]:
                distance = distances[(node, key)]
                current_path_length += distance
                vertices.append((node, key, distance))
                node = key

            graph.remove_vertices()
            graph.add_vertices(vertices)
            if current_path_length < graph.optimal_cycle_length:
                graph.optimal_cycle = ShortestPath(current_path_length, vertices)
                evaluations_until_solved = paths_evaluated
            yield AlgorithmResult(paths_evaluated, evaluations_until_solved)
    graph.remove_vertices()
    graph.add_vertices(graph.optimal_cycle.vertices)
    yield AlgorithmResult(paths_evaluated, evaluations_until_solved)
