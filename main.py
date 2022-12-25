"""An attempt to compare the TSP problem using
brute force/dynamic programming/genetic algorithm/ant colony optimization methods"""
import asyncio
from math import sqrt
import pygame as pg
from aiostream import stream
from graph import Graph, Node
from solvers.bruteforce import brute_force
from solvers.dynamicprogramming import dynamic_programming
from solvers.geneticalgorithm import genetic_algorithm
from solvers.antcolonyoptimization import ant_colony


WIDTH, HEIGHT = 2500, 1300
MARGIN = 10
TEXT_SIZE = 24
NODE_SIZE = 8
(GRAPH_WIDTH, GRAPH_HEIGHT) = GRAPH_SURFACE_DIMENSIONS = ((WIDTH - (5 * MARGIN)) // 4,
                                                          ((HEIGHT - (2 * TEXT_SIZE)
                                                            - (3 * MARGIN)) // 2))
BACKGROUND_COLOR = (0, 0, 0)
TEXT_LABEL_COLOR = (50, 50, 50)
TEXT_LABEL_WORKING_COLOR = (200, 180, 10)
TEXT_LABEL_FINISHED_COLOR = (50, 150, 50)
TEXT_COLOR, TEXT_ALPHA = (250, 250, 250), 175
NODE_COLOR = (10, 25, 195)
VERTEX_COLOR = (150, 150, 150)
BEST_PATH_COLOR = (10, 25, 195)
VERTEX_WIDTH = 2
FPS = 5


def draw_label(screen: pg.surface.Surface, font: pg.font.Font, text: str,
               i: int,color: tuple[int, int, int]) -> None:
    label = pg.Surface((GRAPH_WIDTH, TEXT_SIZE))
    label.fill(color)
    label.set_alpha(TEXT_ALPHA)
    text_element = font.render(text, True, TEXT_COLOR)
    label.blit(text_element, (GRAPH_WIDTH // 2 - text_element.get_width() // 2, 1))
    label_rect = label.get_rect()
    label_rect.topleft = ((MARGIN * i) + (GRAPH_WIDTH * (i-1)),
                          (2 * MARGIN) + TEXT_SIZE + GRAPH_HEIGHT)
    screen.set_clip(label_rect)
    screen.blit(label, label_rect)


def initialize(methods: list[str]) -> tuple[pg.Surface, pg.surface.Surface, pg.surface.Surface,
                                            pg.surface.Surface, pg.surface.Surface,
                                            pg.surface.Surface, pg.font.Font, pg.time.Clock]:
    """Initialize all we need to start running the game"""
    pg.init()
    # pg.event.set_allowed([pg.QUIT, pg.KEYDOWN, pg.MOUSEBUTTONDOWN])
    pg.display.set_caption("Traveling Salesman Problem - comparison of algorithms")
    pg.mouse.set_cursor(pg.SYSTEM_CURSOR_HAND)
    screen = pg.display.set_mode((WIDTH, HEIGHT), pg.RESIZABLE)
    background = pg.image.load("img/background.jpg").convert()
    background = pg.transform.scale(background, (WIDTH, HEIGHT))
    screen.blit(background, (0, 0))

    font = pg.font.Font(pg.font.match_font("calibri"), 24)
    label = pg.Surface((GRAPH_WIDTH, TEXT_SIZE))
    label.fill(TEXT_LABEL_COLOR)
    label.set_alpha(TEXT_ALPHA)
    text = font.render("Draw your set of nodes here", True, TEXT_COLOR)
    label.blit(text, (GRAPH_WIDTH // 2 - text.get_width() // 2, 1))
    screen.blit(label, ((WIDTH // 2) - (GRAPH_WIDTH // 2), MARGIN))
    draw_surface = pg.Surface(GRAPH_SURFACE_DIMENSIONS)
    draw_surface.fill(BACKGROUND_COLOR)
    rect = draw_surface.get_rect()
    rect.topleft = ((WIDTH // 2) - (GRAPH_WIDTH // 2), MARGIN + TEXT_SIZE)
    screen.blit(draw_surface, rect)

    surfaces = [screen, draw_surface]
    for i, method in enumerate(methods, start=1):
        draw_label(screen, font, method, i, TEXT_LABEL_COLOR)
        surface = pg.Surface(GRAPH_SURFACE_DIMENSIONS)
        surface.fill(BACKGROUND_COLOR)
        rect = draw_surface.get_rect()
        rect.topleft = ((GRAPH_WIDTH * (i-1)) + (MARGIN * i),
                        (2 * MARGIN) + GRAPH_HEIGHT + (TEXT_SIZE * 2))
        surfaces.append(surface)
        screen.set_clip(rect)
        screen.blit(surface, rect)
    return (*tuple(surfaces), font, pg.time.Clock())


def handle_events(draw_graph: Graph, running: bool, show_weights: bool
                  )-> tuple[bool, bool, bool]:
    """Handling the PyGame events in the main loop"""
    for event in pg.event.get():
        match event.type:
            case pg.QUIT:
                return running, show_weights, True
            case pg.KEYDOWN:
                match event.key:
                    case pg.K_RETURN:
                        running = not running
                    case pg.K_w:
                        show_weights = not show_weights
            case pg.MOUSEBUTTONDOWN:
                if running:
                    continue
                mouse_button = pg.mouse.get_pressed()
                if mouse_button[0]:
                    mouse_position = pg.mouse.get_pos()
                    x, y = (mouse_position[0] - ((WIDTH // 2) - (GRAPH_WIDTH // 2)),
                            mouse_position[1] - TEXT_SIZE)
                    draw_graph.add_node(Node(len(draw_graph), x, y))
    return running, show_weights, False


def draw_setup_graph(screen: pg.Surface, setup_graph: Graph, setup_surface: pg.surface.Surface):
    setup_surface.fill(BACKGROUND_COLOR)
    for node in setup_graph:
        pg.draw.circle(setup_surface, NODE_COLOR, node.position, NODE_SIZE)
    graph_rect = setup_surface.get_rect()
    graph_rect.topleft = ((WIDTH // 2) - (GRAPH_WIDTH // 2), MARGIN + TEXT_SIZE)
    screen.set_clip(graph_rect)
    screen.blit(setup_surface, graph_rect)


def draw_surfaces(screen: pg.Surface, graphs: list[tuple[Graph, pg.surface.Surface]],
                  show_weights: bool) -> None:
    """Updates all graphs"""
    font_name = pg.font.match_font("calibri")
    font = pg.font.Font(font_name, 12)
    for i, (graph, surface) in enumerate(graphs, start=1):
        surface.fill(BACKGROUND_COLOR)
        for node_key in graph.vertices:
            node = graph[node_key]
            neighbours = graph.vertices[node_key]
            for neighbour_key in neighbours:
                weight = neighbours[neighbour_key]
                other = graph[neighbour_key]
                pg.draw.line(surface, VERTEX_COLOR, node.position, other.position, VERTEX_WIDTH)
                if show_weights:
                    text = font.render(f"{weight}", True, TEXT_COLOR)
                    text_position = tuple(map(lambda x, y: (x + y) // 2,
                                              node.position, other.position))
                    surface.blit(text, text_position)
        for node in graph:
            pg.draw.circle(surface, NODE_COLOR, node.position, NODE_SIZE)
        graph_rect = surface.get_rect()
        graph_rect.topleft = ((GRAPH_WIDTH * (i-1)) + (MARGIN * i),
                              (2 * MARGIN) + GRAPH_HEIGHT + (2 * TEXT_SIZE))
        screen.set_clip(graph_rect)
        screen.blit(surface, graph_rect)


def distance_to(node1: Node, node2: Node) -> int:
    x_distance = node1.x - node2.x
    y_distance = node1.y - node2.y
    return round(sqrt(x_distance * x_distance + y_distance * y_distance))


def calculate_distances(nodes: list[Node]):
    distances: dict[tuple[int, int], int] = {}
    for node in nodes:
        for other in nodes:
            if (node != other and
                (node.key, other.key) not in distances and
                (other.key, node.key) not in distances):
                weight = distance_to(node, other)
                distances[(node.key, other.key)] = weight
                distances[(other.key, node.key)] = weight
    return distances


def update_label(screen: pg.surface.Surface, font: pg.font.Font, shortest_path: int | None,
                 finished: bool | None, methods: list[str], i: int) -> None:
    text = f"{methods[i]}: ({shortest_path})"
    color = TEXT_LABEL_FINISHED_COLOR if finished else TEXT_LABEL_WORKING_COLOR
    draw_label(screen, font, text, i+1, color)


async def solve(screen: pg.Surface, bf_graph: Graph, dyn_graph: Graph, gen_graph: Graph,
                ant_graph: Graph, bf_surface: pg.surface.Surface, dyn_surface: pg.surface.Surface,
                gen_surface: pg.surface.Surface, ant_surface: pg.surface.Surface,
                distances: dict[tuple[int, int], int], clock: pg.time.Clock,
                methods: list[str], show_weights: bool, font: pg.font.Font):
    runs = [brute_force(bf_graph, distances), dynamic_programming(dyn_graph, distances),
            genetic_algorithm(gen_graph, distances), ant_colony(ant_graph, distances)]
    zipped = stream.ziplatest(*runs)
    merged = stream.map(zipped, lambda x: dict(enumerate(x)))
    async with merged.stream() as streamer:
        async for result in streamer:
            print(result)
            if result[0] is not None:
                shortest_path, finished = result[0]
                update_label(screen, font, shortest_path, finished, methods, 0)
            if result[1] is not None:
                shortest_path, finished = result[1]
                update_label(screen, font, shortest_path, finished, methods, 1)
            if result[2] is not None:
                shortest_path, finished = result[2]
                update_label(screen, font, shortest_path, finished, methods, 2)
            if result[3] is not None:
                shortest_path, finished = result[3]
                update_label(screen, font, shortest_path, finished, methods, 3)

            draw_surfaces(screen,[(bf_graph, bf_surface), (dyn_graph, dyn_surface),
                            (gen_graph, gen_surface), (ant_graph, ant_surface)], show_weights)
            pg.display.flip()
            clock.tick(FPS)

async def main():
    """Main function"""
    methods = ["Brute Force", "Dynamic Programming", "Genetic Algorithm", "Ant Colony Optimization"]
    (screen, setup_surface, bf_surface, dyn_surface,
     gen_surface, ant_surface, font, clock) = initialize(methods)
    setup_graph = Graph()
    running = False
    show_weights = True
    distances: dict[tuple[int, int], int] = {}
    while not running:
        running, show_weights, exit_program = handle_events(setup_graph, running, show_weights)

        if exit_program:
            pg.quit()
            break

        draw_setup_graph(screen, setup_graph, setup_surface)
        pg.display.flip()

    nodes = setup_graph.get_nodes()
    if len(nodes) > 1:
        distances = calculate_distances(nodes)
        await loop.create_task(solve(screen, Graph(nodes), Graph(nodes), Graph(nodes), Graph(nodes),
                                     bf_surface, dyn_surface, gen_surface, ant_surface,
                                     distances, clock, methods, show_weights, font))
    while True:
        _ = [exit() for event in pg.event.get() if event.type == pg.QUIT]


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
