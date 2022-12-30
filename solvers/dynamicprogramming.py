"""The dynamic programming algorithm to solve the TSP problem"""
from typing import AsyncIterator
from sys import maxsize

import asyncio

from graph import Graph
from tsptypes import ShortestPath, AlgorithmResult


async def dynamic_programming(graph: Graph, distances: dict[tuple[int, int], int]
                              ) -> AsyncIterator[AlgorithmResult]:
    """Solve the TSP problem with dynamic programming"""
    node_count = len(graph)
    start = graph[0].key
    memo = [[0 for _ in range(1 << node_count)] for __ in range(node_count)]
    optimal_cycle = []
    optimal_cycle_length = maxsize
    cycles_evaluated = 0
    evaluations_until_solved = 0
    memo = setup(memo, graph, distances, start)
    for nodes_in_subcycle in range(3, node_count + 1):
        for subcycle in initialize_combinations(nodes_in_subcycle, node_count):
            if is_not_in(start, subcycle):
                continue
            cycles_evaluated += 1
            # Look for the best next node to attach to the cycle
            for next_node in range(node_count):
                if next_node == start or is_not_in(next_node, subcycle):
                    continue
                subcycle_without_next_node = subcycle ^ (1 << next_node)
                min_cycle_length = maxsize
                for last_node in range(node_count):
                    if (last_node == start or
                        last_node == next_node or
                        is_not_in(last_node, subcycle)):
                        continue
                    new_cycle_length = (memo[last_node][subcycle_without_next_node]
                                        + distances[(last_node, next_node)])
                    if new_cycle_length < min_cycle_length:
                        min_cycle_length = new_cycle_length
                    memo[next_node][subcycle] = min_cycle_length

        evaluations_until_solved = cycles_evaluated
        optimal_cycle_length = calculate_optimal_cycle_length(start, nodes_in_subcycle,
                                                            memo, distances)
        optimal_cycle = find_optimal_cycle(start, nodes_in_subcycle, memo, distances)
        vertices = create_vertices(optimal_cycle, distances)
        graph.optimal_cycle = ShortestPath(optimal_cycle_length, vertices)
        await asyncio.sleep(0.0001)
        yield AlgorithmResult(cycles_evaluated, evaluations_until_solved)

    optimal_cycle_length = calculate_optimal_cycle_length(start, node_count, memo, distances)
    optimal_cycle = find_optimal_cycle(start, node_count, memo, distances)
    vertices = create_vertices(optimal_cycle, distances)
    graph.remove_vertices()
    graph.add_vertices(vertices)
    graph.optimal_cycle = ShortestPath(optimal_cycle_length, vertices)
    yield AlgorithmResult(cycles_evaluated, evaluations_until_solved)


def setup(memo: list, graph: Graph, distances: dict[tuple[int, int], int], start: int) -> list:
    """Prepare the array used for memoization during the dynamic programming algorithm"""
    for i, node in enumerate(graph):
        if start == node.key:
            continue
        memo[i][1 << start | 1 << i] = distances[(start, i)]
    return memo


def initialize_combinations(nodes_in_subcycle: int, node_count: int) -> list[int]:
    """Initialize the combinations to consider in the next step of the algorithm"""
    subcycle_list = []
    initialize_combination(0, 0, nodes_in_subcycle, node_count, subcycle_list)
    return subcycle_list


def initialize_combination(subcycle, at, nodes_in_subcycle, node_count, subcycle_list) -> None:
    """Initialize the combination to consider in the next step of the algorithm"""
    elements_left_to_pick = node_count - at
    if elements_left_to_pick < nodes_in_subcycle:
        return

    if nodes_in_subcycle == 0:
        subcycle_list.append(subcycle)
    else:
        for i in range(at, node_count):
            subcycle |= 1 << i
            initialize_combination(subcycle, i + 1, nodes_in_subcycle - 1,
                                   node_count, subcycle_list)
            subcycle &= ~(1 << i)


def is_not_in(index, subcycle) -> bool:
    """Checks if the bit at the given index is a 0"""
    return ((1 << index) & subcycle) == 0


def calculate_optimal_cycle_length(start: int, node_count: int, memo: list,
                                   distances: dict[tuple[int, int], int]) -> int:
    """Calculate the optimal cycle length"""
    end = (1 << node_count) - 1
    optimal_cycle_length = maxsize
    for i in range(node_count):
        if i == start:
            continue

        cycle_cost = memo[i][end] + distances[(i, start)]
        if cycle_cost < optimal_cycle_length:
            optimal_cycle_length = cycle_cost
    return optimal_cycle_length


def find_optimal_cycle(start: int, node_count: int, memo: list,
                       distances: dict[tuple[int, int], int]):
    """Recreate the optimal cycle"""
    last_index = start
    state = (1 << node_count) - 1
    optimal_cycle: list[int] = []
    for _ in range(node_count - 1, 0, -1):
        index = -1
        for j in range(node_count):
            if j == start or is_not_in(j, state):
                continue
            if index == -1:
                index = j
            prev_cycle_length = memo[index][state] + distances[(index, last_index)]
            new_cycle_length = memo[j][state] + distances[(j, last_index)]
            if new_cycle_length < prev_cycle_length:
                index = j

        optimal_cycle.append(index)
        state = state ^ (1 << index)
        last_index = index
    optimal_cycle.append(start)
    optimal_cycle.reverse()
    optimal_cycle.append(start)
    return optimal_cycle

def create_vertices(optimal_cycle: list[int], distances: dict[tuple[int, int], int]
                    ) -> list[tuple[int, int, int]]:
    """Transform the list of visited node keys to something our graph can work with"""
    vertices: list[tuple[int, int, int]] = []
    for i in range(1, len(optimal_cycle)):
        weight = distances[(optimal_cycle[i-1], optimal_cycle[i])]
        vertices.append((optimal_cycle[i-1], optimal_cycle[i], weight))
    return vertices
