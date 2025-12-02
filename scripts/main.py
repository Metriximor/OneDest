import ijson
import math
import numpy as np
import logging
import json
from uuid import uuid4
from typing import Iterator, Optional, TypeVar
from pydantic import ValidationError
from treelib import Tree
from models import JsonStation, JsonLine, JsonJunctions, JsonJunctionsData
from rustworkx import PyGraph, astar_shortest_path, spring_layout, NoPathFound
from rustworkx.visualization import mpl_draw
from scipy.spatial import KDTree

STATIONS_FILE = "../OneDestStations.json"
LINES_FILE = "../OneDestLines.json"
JUNCTIONS_FILE = "../OneDestJunctions.json"
COLOR_TO_REGION = {
    "#0eaf9b": "lyrean",
    "#d9b5ff": "impendia",
    "#fcff94": "aegis",
    "#8B0000": "northlandia",
    "#534582": "moloka",
    "#024F30": "occident",
    "#fcd20e": "karydia",
    "#ffa500": "founders",
    "#b02e26": "deluvia",
    "#8fd3ff": "arctic",
    "#e83b3b": "medi",
    "#32CD32": "ap",
}


class StreamArray(list):
    ### Class is used to stream a generator object in json.dumps method. Otherwise I get a TypeError: Object of type generator is not JSON serializable
    def __init__(self, iterator: Iterator):
        self.iterator = iterator

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return 1


def euclidean_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


F = TypeVar("F")


def flatten_dict_lists(d: dict[F]) -> list[F]:
    out = []
    for values in d.values():
        out.extend(values)
    return out


def load_lines_to_graph(lines_file: str) -> tuple[PyGraph, dict[tuple[int, int], int]]:
    nodes: dict[tuple[int, int], int] = {}
    lines: list[JsonLine] = []
    graph = PyGraph()

    with open(lines_file, "r") as f:
        for line_json in ijson.items(f, "features.item"):
            try:
                lines.append(JsonLine.model_validate(line_json))
            except ValidationError as e:
                print(e)

    for line_data in lines:
        for polyline in line_data.line:
            for point in polyline:
                if point not in nodes:
                    nodes[point] = graph.add_node(point)
        for polyline in line_data.line:
            for i in range(len(polyline) - 1):
                start, end = polyline[i], polyline[i + 1]
                weight = euclidean_distance(start, end)
                graph.add_edge(nodes[start], nodes[end], weight)
    return graph, nodes


def load_data() -> tuple[dict[str, JsonStation], Tree]:
    stations: dict[str, JsonStation] = {}
    one_dests = Tree()
    with (
        open(STATIONS_FILE, "r") as station_file,
        open(LINES_FILE, "r") as lines_file,
        open(JUNCTIONS_FILE, "r") as junctions_file,
    ):
        one_dests.create_node(identifier="!", tag="/dest !")
        for station_json in ijson.items(station_file, "features.item"):
            station = JsonStation.model_validate(station_json)
            stations[station.name] = station
            parent = "!"
            for dest in station.get_dests():
                if dest in (node.identifier for node in one_dests.children(parent)):
                    parent = dest
                    continue
                one_dests.create_node(identifier=dest, parent=parent)
                parent = dest
    return stations, one_dests


def matchup_stations_to_nodes(
    stations: dict[str, JsonStation], nodes: list[tuple[int, int]]
) -> list[tuple[JsonStation, tuple[int, int]]]:
    node_array = np.array(nodes)
    tree = KDTree(node_array)
    matched: list[tuple[JsonStation, tuple[int, int]]] = []
    for station in stations.values():
        point = (station.x, station.z)
        dist, idx = tree.query(point)
        if dist is None:
            logging.error(f"Failed to match with an edge: {station.model_dump_json()}")
            continue
        nearest_point = tuple(map(int, node_array[idx]))
        if dist > 100:
            logging.error(
                f"{station.name} ({station.x},{station.z}) has no rail nearby, closest is {dist:.2f} blocks away: {nearest_point}"
            )
            continue
        matched.append((station, nearest_point))
    return matched


def group_by_level(
    stations_with_coords: list[tuple[JsonStation, tuple[int, int]]],
) -> list[list[tuple[JsonStation, tuple[int, int]]]]:
    temp: dict[int, list[tuple[JsonStation, tuple[int, int]]]] = {}
    for station, coords in stations_with_coords:
        if station.level not in temp:
            temp[station.level] = []
        temp[station.level].append((station, coords))
    return [group for _, group in sorted(temp.items(), key=lambda item: item[0])]


def pathfind(
    source_coord: tuple[int, int],
    target_coord: tuple[int, int],
    nodes: dict[tuple[int, int], int],
    graph: PyGraph,
) -> Optional[list[tuple[int, int]]]:
    source_idx = nodes[source_coord]
    tx, tz = target_coord

    def goal_fn(node_data: tuple[int, int]):
        x, z = node_data
        return x == tx and z == tz

    def edge_cost_fn(edge_data: tuple[int, int]):
        return edge_data

    def estimate_cost_fn(node_data: tuple[int, int]):
        x, z = node_data
        return euclidean_distance((x, z), (tx, tz))

    try:
        path_indices = astar_shortest_path(
            graph,
            source_idx,
            goal_fn,
            edge_cost_fn,
            estimate_cost_fn,
        )
    except NoPathFound:
        return None

    junctions = []
    for i in path_indices:
        if graph.degree(i) <= 2:
            continue
        junctions.append(graph[i])
    return junctions


def main():
    logging.info("Loading stations and one dest tree")
    unmatched_stations, one_dest_tree = load_data()
    one_dest_tree.save2file("one_dest_tree.txt")
    logging.info("Loading rail graph")
    graph, nodes = load_lines_to_graph(LINES_FILE)
    logging.info("Matching stations to rail graph nodes")
    grouped_stations = group_by_level(
        matchup_stations_to_nodes(unmatched_stations, list(nodes.keys()))
    )
    level_3_stations = grouped_stations[0]
    junctions: dict[tuple[int, int], set[str]] = {}
    i = 0
    for station, coords in level_3_stations:
        for dest_station, dest_coords in level_3_stations:
            if i % 100 == 0:
                logging.info(f"Calculated {i} paths")
            i += 1
            if station == dest_station:
                continue
            path = pathfind(coords, dest_coords, nodes, graph)
            if path is None:
                continue
            for coords in path:
                if coords not in junctions:
                    junctions[coords] = set()
                x, z = coords
                node_quadrant = f"{'+' if x >= 0 else '-'},{'+' if z >= 0 else '-'}"
                dests = dest_station.get_dests()
                quadrant = dests[0]
                region = dests[1]
                nation = dests[2]
                if node_quadrant != quadrant:
                    junctions[coords].add(quadrant)
                    continue
                if COLOR_TO_REGION[station.color] != region:
                    junctions[coords].add(region)
                    continue
                junctions[coords].add(nation)
    with open("../TestJunctions.json", "w+") as junction_test:
        junction_test.write(json.dumps({
            "version": 4,
            "name": "Automated OneDest Switches",
            "source": "local:C75jC5ElD3ER/OneDest Switches",
            "presentations": [{}],
            "features": StreamArray((JsonJunctions(id=str(uuid4()), data=JsonJunctionsData(x=coords[0],z=coords[1],dests=list(junction))).model_dump() for coords, junction in junctions.items())),
        }))

    # positions = circular_layout(graph, center=(0,0))
    print("Drawing graph")
    pos = {
        i: [float(graph[i][0]), float(graph[i][1])] for i in range(graph.num_nodes())
    }
    positions = spring_layout(
        graph,
        pos=pos,
        k=1000.0,
        repulsive_exponent=4,
        num_iter=1000,
        seed=1,
        center=[0.0, 0.0],
    )
    fig = mpl_draw(
        graph,
        node_size=5,
        with_labels=True,
        labels=str,
        font_size=2,
        pos=positions,
        font_color="red",
    )
    ax = fig.gca()
    ax.invert_yaxis()

    fig.savefig(f"graph.png", dpi=300)
    print("Drawn")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s][%(asctime)s][%(filename)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # custom timestamp format
    )
    main()
