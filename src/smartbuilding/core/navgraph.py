"""NavGraph: the navigation graph derived from a Layout. Used by simulator, belief and planners."""

from typing import Any, Self

from pydantic import Field, model_validator

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    DoorId,
    Finite,
    Hash,
    NodeId,
    NodeKind,
    NonNegFloat,
    NonNegInt,
    PosFloat,
)


class NavNode(Part):
    node_id: NodeId
    kind: NodeKind
    floor: NonNegInt = 0
    x_m: Finite
    y_m: Finite
    capacity_pers: NonNegFloat = 0.0
    area_m2: NonNegFloat = 0.0
    region_id: str | None = None  # the Layout region this node stands for

    @model_validator(mode="after")
    def _prefix_matches_kind(self) -> Self:
        if self.node_id.split("_")[1] != self.kind.value:
            raise ValueError(f"node id {self.node_id!r} does not match kind {self.kind.value!r}")
        return self


class NavEdge(Part):
    """Undirected edge. Endpoints are stored in sorted order so equal graphs hash equally."""

    u: NodeId
    v: NodeId
    length_m: PosFloat
    width_m: PosFloat
    accessible: bool = True  # step-free
    stairs: bool = False
    door_id: DoorId | None = None

    @model_validator(mode="before")
    @classmethod
    def _sort_endpoints(cls, data: Any) -> Any:
        if isinstance(data, dict) and "u" in data and "v" in data and data["u"] > data["v"]:
            data = {**data, "u": data["v"], "v": data["u"]}
        return data

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.u == self.v:
            raise ValueError("self-loops are not allowed")
        if self.stairs and self.accessible:
            raise ValueError("a stair edge cannot be marked step-free")
        return self


class NavGraph(Contract):
    """Navigation graph. Node positions are in metres (x east, y north, origin south-west)."""

    derived_from: Hash  # Layout.geometry_hash
    cell_size_m: PosFloat = 0.5
    nodes: tuple[NavNode, ...] = Field(min_length=1)
    edges: tuple[NavEdge, ...] = ()

    def node(self, node_id: str) -> NavNode:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(node_id)

    def nodes_of_kind(self, kind: NodeKind) -> tuple[NavNode, ...]:
        return tuple(n for n in self.nodes if n.kind == kind)

    def adjacency(self) -> dict[str, tuple[tuple[str, NavEdge], ...]]:
        adj: dict[str, list[tuple[str, NavEdge]]] = {n.node_id: [] for n in self.nodes}
        for e in self.edges:
            adj[e.u].append((e.v, e))
            adj[e.v].append((e.u, e))
        return {k: tuple(v) for k, v in adj.items()}

    @model_validator(mode="after")
    def _check_structure(self) -> Self:
        ids = [n.node_id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node ids must be unique")
        known = set(ids)
        seen: set[tuple[str, str]] = set()
        for e in self.edges:
            if e.u not in known or e.v not in known:
                raise ValueError(f"edge {e.u}-{e.v} refers to an unknown node")
            if (e.u, e.v) in seen:
                raise ValueError(f"duplicate edge {e.u}-{e.v}")
            seen.add((e.u, e.v))
        return self
