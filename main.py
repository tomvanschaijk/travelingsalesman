"""An attempt to compare the TSP problem using
brute force/dynamic programming/genetic algorithm/ant colony optimization methods"""
import sys
from math import sqrt, factorial
from typing import AsyncIterator
from enum import Enum

import asyncio
from aiostream import stream
import pygame as pg

from tsptypes import AlgorithmResult
from graph import Graph, Node
from solvers.bruteforce import brute_force
from solvers.dynamicprogramming import dynamic_programming
from solvers.geneticalgorithm import genetic_algorithm
from solvers.antcolonyoptimization import ant_colony


class TSPAlgorithm(Enum):
    """An enum for the algorithms"""
    BF = 0
    DP = 1
    GA = 2
    ACO = 3


WIDTH, HEIGHT = 2500, 1300
MARGIN = 10
TEXT_SIZE = 26
NODE_SIZE = 8
(GRAPH_WIDTH, GRAPH_HEIGHT) = GRAPH_SURFACE_DIMENSIONS = ((WIDTH - (5 * MARGIN)) // 4, ((HEIGHT - (2 * TEXT_SIZE) - (3 * MARGIN)) // 2))
BACKGROUND_COLOR = (0, 0, 0)
TEXT_LABEL_COLOR = (50, 50, 50)
TEXT_LABEL_WORKING_COLOR = (200, 150, 15)
TEXT_LABEL_FINISHED_COLOR = (50, 150, 50)
SETUP_TEXT_COLOR = WEIGHTS_COLOR = (250, 250, 250)
GRAPH_TEXT_COLOR = (0, 0, 0)
NODE_COLOR = (10, 25, 195)
START_NODE_COLOR = (5, 75, 5)
VERTEX_COLOR = (90, 90, 90)
VERTEX_WIDTH = 1
BEST_PATH_VERTEX_COLOR = (50, 150, 50)
BEST_PATH_VERTEX_WIDTH = 3
RESULTS_DISPLAY_DIMENSIONS = (470, 550)
RESULTS_DISPLAY_COLOR, RESULTS_DISPLAY_ALPHA = (150, 150, 150), 175
RESULTS_TEXT_COLOR = (0, 0, 0)
BF_CUTOFF, DP_CUTOFF = 8, 17


class App():
    """Run several algorithms to solve the Traveling Salesman Problem"""
    def __init__(self):
        pg.init()
        pg.display.set_caption("Traveling Salesman Problem - comparison of algorithms")
        pg.mouse.set_cursor(pg.SYSTEM_CURSOR_HAND)
        self.__screen: pg.surface.Surface = pg.display.set_mode((WIDTH, HEIGHT), pg.RESIZABLE)
        self.__background = pg.image.load("img/background.jpg").convert()
        self.__background = pg.transform.scale(self.__background, (WIDTH, HEIGHT))
        font_name = pg.font.match_font("calibri")
        self.__label_font = pg.font.Font(font_name, 24)
        self.__label_font.bold = True
        self.__search_graph_label = pg.Surface((GRAPH_WIDTH, TEXT_SIZE))
        self.__search_graph_label.fill(TEXT_LABEL_COLOR)
        self.__setup_surface = pg.Surface(GRAPH_SURFACE_DIMENSIONS)
        self.__methods = ["Brute Force", "Dynamic Programming",
                          "Genetic Algorithm", "Ant Colony Optimization"]
        self.__weight_font = pg.font.Font(font_name, 12)
        self.__show_weights = True
        self.__results_font = pg.font.Font(font_name, 20)
        self.__setup_graph: Graph
        self.__distances: dict[tuple[int, int], int]
        self.__graphs: tuple[Graph, Graph, Graph, Graph]
        self.__graph_surfaces: tuple[pg.surface.Surface, pg.surface.Surface,
                                     pg.surface.Surface, pg.surface.Surface]
        self.__algorithms: list[AsyncIterator | None]
        self.__running: bool
        self.__finished: bool
        self.__interrupted: bool
        self.__results: dict[int, AlgorithmResult]
        self.__initialize()

    async def run(self):
        """Run the main program loop"""
        while True:
            self.__handle_events()
            if self.__running:
                positions_in_resultset = []
                i = -1
                for algorithm in self.__algorithms:
                    i += 1
                    if algorithm is None:
                        continue

                    positions_in_resultset.append(i)

                algorithms = [algorithm for algorithm in self.__algorithms if algorithm is not None]
                zipped = stream.ziplatest(*algorithms)
                merged = stream.map(zipped, lambda x: dict(enumerate(x)))
                async with merged.stream() as streamer:
                    async for resultset in streamer:
                        self.__handle_events()
                        if self.__interrupted:
                            break

                        i = 0
                        for key in resultset:
                            result = resultset[key]
                            if result is None:
                                i += 1
                                continue

                            self.__results[positions_in_resultset[i]] = result
                            i += 1

                        if not self.__finished:
                            self.__draw_setup_graph()
                            self.__update_labels()
                            self.__draw_graphs()
                            pg.display.flip()

                self.__finished = True
                self.__running = False
                if self.__interrupted:
                    self.__initialize()
                else:
                    self.__update_labels()
                    self.__show_results()
            else:
                self.__draw_setup_graph()
            pg.display.flip()

    def __initialize(self):
        """Initialize all we need to start running the game"""
        self.__setup_graph = Graph()
        self.__graphs = (Graph(),) * len(self.__methods)
        self.__algorithms = [None] * len(self.__methods)
        self.__running = False
        self.__finished = False
        self.__interrupted = False
        self.__results = {}
        self.__distances = {}
        self.__reset_screen()

    def __reset_screen(self):
        """Redraw the entire screen"""
        rect = self.__screen.get_rect()
        self.__screen.set_clip(rect)
        self.__setup_surface.fill(BACKGROUND_COLOR)
        text_element = self.__label_font.render("Search Graph", True, SETUP_TEXT_COLOR)
        self.__search_graph_label.blit(text_element, (GRAPH_WIDTH // 2 - text_element.get_width() // 2, 1))
        self.__screen.blit(self.__background, (0, 0))
        self.__screen.blit(self.__search_graph_label, ((WIDTH // 2) - (GRAPH_WIDTH // 2), MARGIN))
        rect.topleft = ((WIDTH // 2) - (GRAPH_WIDTH // 2), MARGIN + TEXT_SIZE)
        self.__screen.blit(self.__setup_surface, rect)
        surfaces = []
        for i, method in enumerate(self.__methods, start=1):
            self.__draw_label(method, i, TEXT_LABEL_COLOR)
            surface = pg.Surface(GRAPH_SURFACE_DIMENSIONS)
            surface.fill(BACKGROUND_COLOR)
            rect = self.__setup_surface.get_rect()
            rect.topleft = ((GRAPH_WIDTH * (i - 1)) + (MARGIN * i), (2 * MARGIN) + GRAPH_HEIGHT + (TEXT_SIZE * 2))
            surfaces.append(surface)
            self.__screen.blit(surface, rect)
        self.__graph_surfaces = tuple(surfaces)
        self.__draw_graphs()

    def __draw_label(self, text: str, i: int, color: tuple[int, int, int]) -> None:
        """Draw a label above one of the graphs"""
        label = pg.Surface((GRAPH_WIDTH, TEXT_SIZE))
        label.fill(color)
        text_element = self.__label_font.render(text, True, GRAPH_TEXT_COLOR)
        label.blit(text_element, (GRAPH_WIDTH // 2 - text_element.get_width() // 2, 1))
        label_rect = label.get_rect()
        label_rect.topleft = ((MARGIN * i) + (GRAPH_WIDTH * (i - 1)), (2 * MARGIN) + TEXT_SIZE + GRAPH_HEIGHT)
        self.__screen.set_clip(label_rect)
        self.__screen.blit(label, label_rect)

    def __draw_setup_graph(self) -> None:
        """Draw the setup graph"""
        self.__setup_surface.fill(BACKGROUND_COLOR)
        for i, node in enumerate(self.__setup_graph):
            color = START_NODE_COLOR if i == 0 else NODE_COLOR
            pg.draw.circle(self.__setup_surface, color, node.position, NODE_SIZE)
        graph_rect = self.__setup_surface.get_rect()
        graph_rect.topleft = ((WIDTH // 2) - (GRAPH_WIDTH // 2), MARGIN + TEXT_SIZE)
        self.__screen.set_clip(graph_rect)
        self.__screen.blit(self.__setup_surface, graph_rect)

    def __update_labels(self) -> None:
        """Update all algorithm graph labels"""
        for i, graph in enumerate(self.__graphs):
            optimal_cycle_length = ("unknown" if graph.optimal_cycle_length == sys.maxsize
                                    else f"{graph.optimal_cycle_length:n}")
            text = f"{self.__methods[i]} (best path length: {optimal_cycle_length})"
            color = (TEXT_LABEL_COLOR if optimal_cycle_length == "unknown"
                     else TEXT_LABEL_FINISHED_COLOR if self.__finished
                     else TEXT_LABEL_WORKING_COLOR)
            self.__draw_label(text, i + 1, color)

    def __draw_graphs(self) -> None:
        """Draws all algorithm graphs on their respective surfaces"""
        for i, (graph, graph_surface) in enumerate(zip(self.__graphs, self.__graph_surfaces), start=1):
            graph_surface.fill(BACKGROUND_COLOR)
            self.__draw_vertices(graph, graph_surface)
            self.__draw_optimal_path(graph, graph_surface)
            self.__draw_nodes(graph, graph_surface)
            graph_rect = graph_surface.get_rect()
            graph_rect.topleft = ((GRAPH_WIDTH * (i - 1)) + (MARGIN * i), (2 * MARGIN) + GRAPH_HEIGHT + (2 * TEXT_SIZE))
            self.__screen.set_clip(graph_rect)
            self.__screen.blit(graph_surface, graph_rect)

    def __draw_vertices(self, graph: Graph, graph_surface: pg.surface.Surface) -> None:
        """Draw graph vertices"""
        for node_key in graph.vertices:
            node = graph[node_key]
            neighbours = graph.vertices[node_key]
            for neighbour_key in neighbours:
                weight = neighbours[neighbour_key]
                neighbour = graph[neighbour_key]
                self.__draw_vertex(node, neighbour, weight, VERTEX_COLOR, VERTEX_WIDTH, graph_surface)

    def __draw_optimal_path(self, graph: Graph, graph_surface: pg.surface.Surface) -> None:
        """Draw the vertices of the optimal path in a graph"""
        for node_key, neighbour_key, weight in graph.optimal_cycle.vertices:
            node = graph[node_key]
            neighbour = graph[neighbour_key]
            self.__draw_vertex(node, neighbour, weight, BEST_PATH_VERTEX_COLOR, BEST_PATH_VERTEX_WIDTH, graph_surface)

    def __draw_vertex(self, node: Node, neighbour: Node, weight: int,
                      vertex_color: tuple[int, int, int], vertex_width: int,
                      graph_surface: pg.surface.Surface) -> None:
        pg.draw.line(graph_surface, vertex_color, node.position, neighbour.position, vertex_width)
        if self.__show_weights:
            text = self.__weight_font.render(f"{weight}", True, WEIGHTS_COLOR)
            text_position = tuple(map(lambda x, y: (x + y) // 2, node.position, neighbour.position))
            text_rect = text.get_rect()
            text.set_clip(text_rect)
            graph_surface.blit(text, text_position)

    def __draw_nodes(self, graph: Graph, graph_surface: pg.surface.Surface) -> None:
        """Draw the nodes in a graph"""
        for i, node in enumerate(graph):
            color = START_NODE_COLOR if i == 0 else NODE_COLOR
            pg.draw.circle(graph_surface, color, node.position, NODE_SIZE)

    def __calculate_distances(self, nodes: list[Node]) -> None:
        """Calculate the distance between all nodes in the setup graph"""
        for node in nodes:
            for other in nodes:
                if (node != other and (node.key, other.key) not in self.__distances and (other.key, node.key) not in self.__distances):
                    weight = self.__distance_to(node, other)
                    self.__distances[(node.key, other.key)] = weight
                    self.__distances[(other.key, node.key)] = weight

    def __distance_to(self, node1: Node, node2: Node) -> int:
        """Calculate the Euclidean distance between 2 points in the graph"""
        x_distance = node1.pos_x - node2.pos_x
        y_distance = node1.pos_y - node2.pos_y
        return round(sqrt(x_distance * x_distance + y_distance * y_distance))

    def __show_results(self) -> None:
        """Shows the results for all algorithms"""
        results_display = self.__create_results_display()
        results_display_rect = results_display.get_rect()
        results_display_rect.topleft = (MARGIN, MARGIN)
        self.__screen.set_clip(results_display_rect)
        self.__screen.blit(results_display, results_display_rect)

    def __create_results_display(self) -> pg.Surface:
        """Creates the display for the algorithm results"""
        results_display = pg.Surface(RESULTS_DISPLAY_DIMENSIONS)
        results_display.fill(RESULTS_DISPLAY_COLOR)
        results_display.set_alpha(RESULTS_DISPLAY_ALPHA)
        node_count = len(self.__setup_graph)
        path_count = factorial(node_count)
        possible_paths = f"{path_count:n}" if len(str(path_count)) < 8 else f"{path_count:.2e}"
        text = (f"Results for {node_count} nodes " + f"({possible_paths} possible paths):")
        text_surface = self.__results_font.render(text, True, RESULTS_TEXT_COLOR)
        results_display.blit(text_surface, (25, 10))
        result_texts = [
            ("generations evaluated", "generations until best approximation"),
            ("swarms evaluated", "evaluations until best approximation"),
            ("paths evaluated", "evaluations until solved"),
            ("subcycles evaluated", "evaluations until solved")
        ]

        j = -1
        for i, method in enumerate(self.__methods):
            algorithm = TSPAlgorithm(i).value
            if algorithm not in self.__results:
                continue

            j += 1
            result = self.__results[i]
            if result is not None:
                text_surface = self.__results_font.render(method, True, RESULTS_TEXT_COLOR)
                results_display.blit(text_surface, (25, 50 + (j * 125)))
                text = f"{result_texts[i][0]}: {result.count_evaluated:n}"
                text_surface = self.__results_font.render(text, True, RESULTS_TEXT_COLOR)
                results_display.blit(text_surface, (50, 50 + (j * 125) + 25))
                text = f"{result_texts[i][1]}: {result.evaluations_until_solved:n}"
                text_surface = self.__results_font.render(text, True, RESULTS_TEXT_COLOR)
                results_display.blit(text_surface, (50, 50 + (j * 125) + 50))
        return results_display

    def __determine_ga_parameters(self, node_count: int) -> tuple[int, int]:
        match node_count:
            case _ if node_count <= 5: return 50, 5
            case _ if node_count <= 10: return 250, 10
            case _ if node_count <= 15: return 500, 30
            case _ if node_count <= 20: return 750, 50
            case _ if node_count <= 25: return 1000, 75
            case _ if node_count <= 30: return 1250, 100
            case _ if node_count <= 35: return 1500, 150
            case _ if node_count <= 50: return 5000, 250
        return 25000, 500

    def __handle_events(self) -> None:
        """Handling the PyGame events in the main loop"""
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    pg.quit()
                    sys.exit()
                case pg.KEYDOWN:
                    match event.key:
                        case pg.K_SPACE:
                            if self.__running:
                                self.__interrupted = True
                            else:
                                self.__initialize()
                        case pg.K_RETURN | pg.K_KP_ENTER:
                            if not self.__running:
                                self.__reset_screen()
                                nodes = self.__setup_graph.get_nodes()
                                if len(nodes) <= 2:
                                    continue

                                self.__graphs = (Graph(nodes.copy()), Graph(nodes.copy()), Graph(nodes.copy()), Graph(nodes.copy()))
                                self.__calculate_distances(nodes)
                                (pop_size, max_generations) = self.__determine_ga_parameters(len(self.__setup_graph))
                                node_count = len(nodes)
                                self.__algorithms = [None] * len(self.__methods)
                                if node_count <= BF_CUTOFF:
                                    self.__algorithms[0] = brute_force(self.__graphs[0], self.__distances)
                                if node_count <= DP_CUTOFF:
                                    self.__algorithms[1] = dynamic_programming(self.__graphs[1], self.__distances)

                                self.__algorithms[2] = genetic_algorithm(self.__graphs[2], self.__distances, pop_size, max_generations, 50)
                                self.__algorithms[3] = ant_colony(self.__graphs[3], self.__distances, 50, 5)
                                self.__finished = False
                                self.__interrupted = False
                                self.__running = not self.__running
                                self.__results = {}
                        case pg.K_w:
                            self.__show_weights = not self.__show_weights
                case pg.MOUSEBUTTONDOWN:
                    if self.__running:
                        continue
                    mouse_button = pg.mouse.get_pressed()
                    if mouse_button[0]:
                        mouse_x, mouse_y = pg.mouse.get_pos()
                        if (((WIDTH // 2) - (GRAPH_WIDTH // 2)) < mouse_x < ((WIDTH // 2) + (GRAPH_WIDTH // 2)) and (MARGIN + TEXT_SIZE) < mouse_y < (MARGIN + TEXT_SIZE + GRAPH_HEIGHT)):
                            node_x, node_y = (mouse_x - ((WIDTH // 2) - (GRAPH_WIDTH // 2)), mouse_y - TEXT_SIZE)
                            self.__setup_graph.add_node(Node(len(self.__setup_graph), node_x, node_y))


if __name__ == "__main__":
    app = App()
    asyncio.run(app.run())
