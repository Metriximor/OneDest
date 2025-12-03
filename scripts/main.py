import ijson
import math
import numpy as np
import logging
import json
from uuid import uuid4
from typing import Iterable, Iterator, Optional, TypeVar
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
    "#e83b3b": "medi",
    "#00a1de": "arctic",
    "#3FBE23": "ap",
    "#8c100a": "deluvia",
    "#41EADF": "karydia",
    "#013220": "occident",
    "#ffa500": "founders",
    "#8B0000": "northlandia",
    "#4e09ed": "moloka",
    "#ff00f3": "impendia",
    "#fcd20e": "aegis",
    "#ff5e00": "lyrean"
}


class StreamArray(list):
    ### Class is used to stream a generator object in json.dumps method. Otherwise I get a TypeError: Object of type generator is not JSON serializable
    def __init__(self, iterator: Iterator):
        self.iterator = iterator

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return 1


T = TypeVar("T")


def pairwise_with_tail(iterable: Iterable[T]) -> Iterator[tuple[T, Optional[T]]]:
    it = iter(iterable)
    try:
        prev: T = next(it)
    except StopIteration:
        return  # empty iterable → yields nothing

    for curr in it:
        yield prev, curr
        prev = curr

    yield prev, None


def euclidean_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


F = TypeVar("F")


def flatten_dict_lists(d: dict[F]) -> list[F]:
    out = []
    for values in d.values():
        out.extend(values)
    return out


def load_lines_to_graph(
    lines_file: str,
) -> tuple[PyGraph, dict[tuple[int, int], int], dict[tuple[int, int], list[JsonLine]]]:
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

    junction_to_lines: dict[tuple[int, int], list[JsonLine]] = {}
    for line_data in lines:
        for polyline in line_data.line:
            for point in polyline:
                if point not in junction_to_lines:
                    junction_to_lines[point] = []
                junction_to_lines[point].append(line_data)
    return graph, nodes, junction_to_lines


def load_stations_data() -> tuple[dict[str, JsonStation], Tree]:
    stations: dict[str, JsonStation] = {}
    one_dests = Tree()
    with (open(STATIONS_FILE, "r") as station_file,):
        one_dests.create_node(identifier="!", tag="/dest !")
        for station_json in ijson.items(station_file, "features.item"):
            station = JsonStation.model_validate(station_json)
            stations[station.name] = station
            parent = "!"
            for dest in station.get_dests():
                id = f"{parent}/{dest}"
                if id in (node.identifier for node in one_dests.children(parent)):
                    parent = id
                    continue
                one_dests.create_node(identifier=id, tag=dest, parent=parent)
                parent = id
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

    return path_indices


def match_junctions_to_line(
    starting: tuple[int, int],
    ending: tuple[int, int],
    junctions_to_lines: dict[tuple[int, int], list[JsonLine]],
) -> JsonLine:
    starting_junction_lines = junctions_to_lines[starting]
    ending_junction_lines = junctions_to_lines[ending]
    matches = set()
    for line in starting_junction_lines:
        if line in ending_junction_lines:
            matches.add(line.id)
    matches = list(matches)

    if len(matches) == 0:
        raise ValueError("No junction found")
    if len(matches) > 1:
        raise ValueError(
            f"More than one junction found for {starting} and {ending}: {matches}"
        )
    for line in starting_junction_lines:
        if line.id == matches[0]:
            return line


def main():
    logging.info("Loading stations and one dest tree")
    unmatched_stations, one_dest_tree = load_stations_data()
    one_dest_tree.save2file("one_dest_tree.txt")
    logging.info("Loading rail graph")
    graph, nodes, junction_to_lines = load_lines_to_graph(LINES_FILE)
    logging.info("Matching stations to rail graph nodes")
    grouped_stations = matchup_stations_to_nodes(unmatched_stations, list(nodes.keys()))
    junctions = {}
    cnt = 0
    highest_level = max(station.level for station, _ in grouped_stations)
    for level in range(3, highest_level+1):
        logging.info(f"Processing level {level} routes")
        for station, origin_coords in grouped_stations:
            for dest_station, dest_coords in grouped_stations:
                if station == dest_station:
                    continue
                if station.level != level or dest_station.level != level:
                    continue
                if cnt % 500 == 0:
                    logging.info(f"Calculated {cnt} paths")
                cnt += 1
                path = pathfind(origin_coords, dest_coords, nodes, graph)
                if path is None:
                    # logging.error(
                    #     f"Couldn't pathfind to {dest_station.name} from {station.name}"
                    # )
                    continue
                for i in range(len(path) - 1):
                    coords = graph[path[i]]
                    if graph.degree(path[i]) <= 2:
                        continue
                    if coords not in junctions:
                        junctions[coords] = {}
                    x, z = coords
                    node_quadrant = f"{'+' if x >= 0 else '-'},{'+' if z >= 0 else '-'}"
                    dests = dest_station.get_dests()
                    quadrant = dests[0]
                    region = dests[1]
                    nation = dests[2]
                    successor = graph[path[i + 1]]
                    angle = math.degrees(math.atan2(-(successor[1] - z), (successor[0] - x)))
                    line = match_junctions_to_line(coords, successor, junction_to_lines)
                    node_region = COLOR_TO_REGION[line.color]
                    if angle not in junctions[coords]:
                        junctions[coords][angle] = set()
                    current_junction_dests = set()
                    for junction_angle in junctions[coords]:
                        current_junction_dests |= junctions[coords][junction_angle]
                    
                    # for the cases where a region goes across quadrant boundaries
                    if node_quadrant != quadrant and node_region == region:
                        if nation in current_junction_dests and nation not in junctions[coords][angle]:
                            junctions[coords][angle].add(dests[3])
                            continue
                        junctions[coords][angle].add(nation)
                        continue
                    if node_quadrant != quadrant:
                        if quadrant in current_junction_dests and quadrant not in junctions[coords][angle]:
                            junctions[coords][angle].add(region)
                            continue
                        junctions[coords][angle].add(quadrant)
                        continue
                    if node_region != region:
                        if region in current_junction_dests and region not in junctions[coords][angle]:
                            junctions[coords][angle].add(nation)
                            continue
                        junctions[coords][angle].add(region)
                        continue
                    if nation in current_junction_dests and nation not in junctions[coords][angle]:
                        junctions[coords][angle].add(dests[3])
                        continue
                    junctions[coords][angle].add(nation)
    # Map the set back into a list so I can dump it into a json
    output = {}
    for coord, junction in junctions.items():
        output[coord] = {}
        for id, dest_set in junction.items():
            output[coord][f"{id:.2f}"] = list(dest_set)
    with open("../TestJunctions.json", "w+") as junction_test:
        # junction_test.write(json.dumps(output))
        junction_dump = StreamArray(
            JsonJunctions(
                id=str(uuid4()),
                data=JsonJunctionsData(x=coords[0], z=coords[1], dests=junction),
            ).model_dump()
            for coords, junction in output.items()
        )
        junction_test.write(
            json.dumps(
                {
                    "version": 4,
                    "name": "Automated OneDest Switches",
                    "source": "local:C75jC5ElD3ER/OneDest Switches",
                    "presentations": [{}],
                    "features": junction_dump,
                }
            )
        )

    # positions = circular_layout(graph, center=(0,0))
    print("Drawing graph")
    pos = {i: graph[i] for i in range(graph.num_nodes())}
    positions = spring_layout(
        graph,
        pos=pos,
        k=1000.0,
        repulsive_exponent=4,
        num_iter=1000,
        seed=1,
        center=(0.0, 0.0),
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
    assert fig is not None
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
