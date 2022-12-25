from sys import maxsize
from itertools import permutations
import asyncio
from graph import Graph


async def brute_force(graph: Graph, distances: dict[tuple[int, int], int]):
    shortest_path = maxsize
    unique_permutations = set()
    for i, permutation in enumerate(permutations(graph.nodes.keys())):
        if i % 500 == 0:
            await asyncio.sleep(0.001)
        if permutation not in unique_permutations and permutation[::-1] not in unique_permutations:
            unique_permutations.add(permutation)
            current_pathweight = 0
            node = permutation[0]
            vertices: list[tuple[int, int, int]] = []
            for key in permutation[1:]:
                distance = distances[(node, key)]
                current_pathweight += distance
                vertices.append((node, key, distance))
                node = key
            current_pathweight += distances[(node, permutation[0])]
            distance = distances[(node, permutation[0])]
            vertices.append((node, permutation[0], distance))

            if shortest_path > current_pathweight:
                shortest_path = current_pathweight
                graph.remove_vertices()
                graph.add_vertices(vertices)

            yield shortest_path, False
    yield shortest_path, True
