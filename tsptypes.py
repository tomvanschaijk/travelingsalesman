"""Helper types for the app"""
from sys import maxsize
from typing import NamedTuple


class ShortestPath(NamedTuple):
    """A tuple that express the shortest path in a graph"""
    length: int = maxsize
    vertices: list[tuple[int, int, int]] = []


class AlgorithmResult(NamedTuple):
    """A tuple that express the results of a TSP algorithm"""
    count_evaluated: int = 0
    evaluations_until_solved: int = 0
