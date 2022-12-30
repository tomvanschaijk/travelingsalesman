"""Types to model a graph"""

from tsptypes import ShortestPath


class Node():
    """A node in a graph"""
    def __init__(self, key: int, pos_x: int, pos_y: int):
        self.__key = key
        self.__pos_x = pos_x
        self.__pos_y = pos_y

    @property
    def key(self) -> int:
        """The unique key for the node"""
        return self.__key

    @property
    def pos_x(self) -> int:
        """The x position of the node"""
        return self.__pos_x

    @property
    def pos_y(self) -> int:
        """The y position of the node"""
        return self.__pos_y

    @property
    def position(self) -> tuple[int, int]:
        """The position of the node"""
        return (self.__pos_x, self.__pos_y)

    def __eq__(self, other):
        return (self.__key == other.key and
                self.__pos_x == other.pos_x and
                self.__pos_y == other.pos_y)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"{self.__key}: ({self.__pos_x}, {self.__pos_y})"

    def __hash__(self) -> int:
        return hash(self.__key)


class Graph():
    """A class for an undirected weighted graph"""
    def __init__(self, nodes: list[Node] = []):
        self.__nodes: dict[int, Node] = {node.key: node for node in nodes}
        self.__vertices: dict[int, dict[int, int]] = {}
        self.__optimal_cycle = ShortestPath()

    @property
    def nodes(self) -> dict[int, Node]:
        """A dictionary of nodes"""
        return self.__nodes

    @property
    def vertices(self) -> dict[int, dict[int, int]]:
        """A dictionary of vertices"""
        return self.__vertices

    @property
    def optimal_cycle(self) -> ShortestPath:
        """The cycle with the shortest cycle in the graph"""
        return self.__optimal_cycle

    @optimal_cycle.setter
    def optimal_cycle(self, value: ShortestPath):
        self.__optimal_cycle = value

    @property
    def optimal_cycle_length(self) -> int:
        """The length of the shortest path in the graph"""
        return self.__optimal_cycle.length

    def get_nodes(self) -> list[Node]:
        """Returns the list of nodes in the graph"""
        return list(self.__nodes.values())

    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.__nodes[node.key] = node

    def add_vertex(self, node: int, neighbour: int, weight: int):
        """Adds a vertex to the graph"""
        if node not in self.__nodes:
            return
        if neighbour not in self.__nodes:
            return
        if node not in self.__vertices:
            self.__vertices[node] = {}
        self.__vertices[node][neighbour] = weight

    def add_vertices(self, vertices: list[tuple[int, int, int]]):
        """Adds a collection of vertices to the graph"""
        _ = [self.add_vertex(node, neighbour, weight) for node, neighbour, weight in vertices]

    def remove_vertex(self, node: int, neighbour: int) -> None:
        """Removes the vertext between the given nodes"""
        if node not in self.__vertices:
            return
        if neighbour not in self.__vertices:
            return
        if neighbour not in self.__vertices[node]:
            return
        self.__vertices[node].pop(neighbour, None)

    def remove_vertices(self) -> None:
        """Removes all vertices"""
        self.__vertices = {}

    def __contains__(self, node: int):
        return node in self.__nodes

    def __getitem__(self, index):
        return self.__nodes[index]

    def __iter__(self):
        for node in self.__nodes:
            yield self.__nodes[node]

    def __len__(self):
        return len(self.__nodes)

    def __del__(self):
        self.__nodes = {}
        self.__vertices = {}
