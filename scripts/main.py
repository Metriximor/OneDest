import ijson
import math
import numpy as np
import logging
from typing import Optional, OrderedDict, TypeVar
from pydantic import ValidationError
from treelib import Tree
from models import JsonStation, JsonLine
from rustworkx import PyGraph, astar_shortest_path
from rustworkx.visualization import mpl_draw
from scipy.spatial import KDTree

STATIONS_FILE = "../OneDestStations.json"
LINES_FILE = "../OneDestLines.json"
JUNCTIONS_FILE = "../OneDestJunctions.json"


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
) -> list[list[int, tuple[JsonStation, tuple[int, int]]]]:
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

    path_indices = astar_shortest_path(
        graph,
        source_idx,
        goal_fn,
        edge_cost_fn,
        estimate_cost_fn,
    )

    junctions = []
    for i in path_indices:
        if graph.degree(i) <= 2:
            continue
        junctions.append(graph[i])
    if len(junctions) == 0:
        return None
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
    l3_stations = grouped_stations[0]
    stationA, stationACoords = l3_stations[0]
    stationB, stationBCoords = next(
        filter(lambda x: x[0].name == "Khalkedonia", l3_stations)
    )
    print(
        f"To go from {stationA.name} to {stationB.name} you must go through: {pathfind(stationACoords, stationBCoords, nodes, graph)}"
    )

    positions = {i: graph[i] for i in range(graph.num_nodes())}
    fig = mpl_draw(
        graph, node_size=5, with_labels=True, labels=str, font_size=2, pos=positions
    )
    ax = fig.gca()
    ax.invert_yaxis()

    fig.savefig("graph.png", dpi=300)
    print("Drawn")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s][%(asctime)s][%(filename)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # custom timestamp format
    )
    main()
