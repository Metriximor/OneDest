from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


class Coordinate(BaseModel):
    x: int
    z: int
    y: int = 0


class JsonLine(BaseModel):
    id: str
    line: list[list[tuple[int, int]]]
    color: str


class JsonStation(BaseModel):
    name: str
    x: int
    z: int
    dest: str
    level: int
    zoom: int
    id: str
    color: str
    y: int
    image: Optional[str] = None

    def get_dests(self) -> list[str]:
        return self.dest.strip().lower().removeprefix("/dest ! ").split(" ")


class JsonJunctionsData(BaseModel):
    x: int
    z: int
    dests: Any
    radius: int = 10


class JsonJunctions(BaseModel):
    id: str
    data: JsonJunctionsData
