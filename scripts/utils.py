import math
from typing import Iterator, Optional, TypeVar


def find_angle_between(point: tuple[int, int], other: tuple[int, int]) -> int:
    """Correctly inverts the Y just like minecraft does"""
    x, z = point
    ox, oz = other
    angle = math.degrees(math.atan2(-(oz - z), (ox - x)))
    if angle == -180:
        return 180
    return angle


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


T = TypeVar("T")


def safe_list_access(
    list: list[T], index: int, default: Optional[T] = None
) -> Optional[T]:
    try:
        return list[index]
    except IndexError:
        return default
