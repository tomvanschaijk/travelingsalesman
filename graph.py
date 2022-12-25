class Node():
    def __init__(self, key: int, x: int, y: int):
        self.__key = key
        self.__x = x
        self.__y = y

    @property
    def key(self) -> int:
        return self.__key

    @property
    def x(self) -> int:
        return self.__x

    @property
    def y(self) -> int:
        return self.__y

    @property
    def position(self) -> tuple[int, int]:
        return (self.__x, self.__y)

    def __eq__(self, other):
        return (self.__key == other.key and
                self.__x == other.x and
                self.__y == other.y)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"{self.__key}: ({self.__x}, {self.__y})"

    def __hash__(self) -> int:
        return hash(self.__key)


class Graph():
    def __init__(self, nodes: list[Node] = []):
        self.__nodes: dict[int, Node] = {node.key: node for node in nodes}
        self.__vertices: dict[int, dict[int, int]] = {}

    @property
    def nodes(self) -> dict[int, Node]:
        return self.__nodes

    @property
    def vertices(self) -> dict[int, dict[int, int]]:
        return self.__vertices

    def get_nodes(self) -> list[Node]:
        return list(self.__nodes.values())

    def add_node(self, node: Node) -> None:
        self.__nodes[node.key] = node

    def add_vertex(self, node: int, neighbour: int, weight: int):
        if node not in self.__nodes:
            return
        if neighbour not in self.__nodes:
            return
        if node not in self.__vertices:
            self.__vertices[node] = {}
        self.__vertices[node][neighbour] = weight

    def add_vertices(self, vertices: list[tuple[int, int, int]]):
        _ = [self.add_vertex(node, neighbour, weight) for node, neighbour, weight in vertices]

    def remove_vertex(self, node: int, neighbour: int) -> None:
        if node not in self.__vertices:
            return
        if neighbour not in self.__vertices:
            return
        if neighbour not in self.__vertices[node]:
            return
        self.__vertices[node].pop(neighbour, None)

    def remove_vertices(self) -> None:
        self.__vertices = {}

    def clear(self) -> None:
        self.__nodes = {}
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
