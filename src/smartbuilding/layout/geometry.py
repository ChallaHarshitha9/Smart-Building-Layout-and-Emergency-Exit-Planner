"""Grid geometry on integer cells. Pure functions, no randomness.

Boundary conventions (see docs/conventions.md): a *boundary line* is a grid line. A door or
exit opening sits on a grid line and spans ``width_cells`` cells along it. Horizontal lines
have constant y and span along x; vertical lines have constant x and span along y.
"""

from dataclasses import dataclass

from smartbuilding.core.layout import Door, Exit, Rect
from smartbuilding.core.primitives import Side


def area_m2(rect: Rect, cell_size_m: float) -> float:
    return rect.area_cells * cell_size_m * cell_size_m


def centroid_m(rect: Rect, cell_size_m: float) -> tuple[float, float]:
    return ((rect.x + rect.w / 2) * cell_size_m, (rect.y + rect.d / 2) * cell_size_m)


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def overlap_cells(a: Rect, b: Rect) -> int:
    """Area (in cells) shared by two rectangles; 0 if they only touch or are apart."""
    w = min(a.x2, b.x2) - max(a.x, b.x)
    d = min(a.y2, b.y2) - max(a.y, b.y)
    return w * d if w > 0 and d > 0 else 0


def inside_grid(rect: Rect, grid_w: int, grid_d: int) -> bool:
    return rect.x2 <= grid_w and rect.y2 <= grid_d


@dataclass(frozen=True)
class Boundary:
    """A shared boundary segment between two rectangles."""

    orientation: str  # "h" (constant y, spans x) or "v" (constant x, spans y)
    line: int
    lo: int
    hi: int

    @property
    def length_cells(self) -> int:
        return self.hi - self.lo


def shared_boundary(a: Rect, b: Rect) -> Boundary | None:
    """The segment where two rectangles touch, or None. Overlapping rectangles may touch too."""
    if a.y2 == b.y or b.y2 == a.y:
        line = a.y2 if a.y2 == b.y else b.y2
        lo, hi = max(a.x, b.x), min(a.x2, b.x2)
        if hi > lo:
            return Boundary("h", line, lo, hi)
    if a.x2 == b.x or b.x2 == a.x:
        line = a.x2 if a.x2 == b.x else b.x2
        lo, hi = max(a.y, b.y), min(a.y2, b.y2)
        if hi > lo:
            return Boundary("v", line, lo, hi)
    return None


def exterior_contacts(rect: Rect, grid_w: int, grid_d: int) -> dict[Side, int]:
    """Contact length (cells) of a rectangle with each outer boundary of the footprint."""
    out: dict[Side, int] = {}
    if rect.x == 0:
        out[Side.west] = rect.d
    if rect.x2 == grid_w:
        out[Side.east] = rect.d
    if rect.y == 0:
        out[Side.south] = rect.w
    if rect.y2 == grid_d:
        out[Side.north] = rect.w
    return out


def door_fits_boundary(door: Door, boundary: Boundary | None) -> bool:
    """True if the door opening lies on the shared boundary and within its extent."""
    if boundary is None:
        return False
    if boundary.orientation == "h":
        return (
            door.y == boundary.line
            and door.x >= boundary.lo
            and door.x + door.width_cells <= boundary.hi
        )
    return (
        door.x == boundary.line
        and door.y >= boundary.lo
        and door.y + door.width_cells <= boundary.hi
    )


def door_center_m(door: Door, boundary: Boundary, cell_size_m: float) -> tuple[float, float]:
    half = door.width_cells / 2
    if boundary.orientation == "h":
        return ((door.x + half) * cell_size_m, boundary.line * cell_size_m)
    return (boundary.line * cell_size_m, (door.y + half) * cell_size_m)


def exit_side(exit_: Exit, grid_w: int, grid_d: int) -> Side | None:
    """Which outer wall the exit is on (declared side if given, else inferred)."""
    if exit_.side is not None:
        return exit_.side
    if exit_.x == 0:
        return Side.west
    if exit_.x == grid_w:
        return Side.east
    if exit_.y == 0:
        return Side.south
    if exit_.y == grid_d:
        return Side.north
    return None


def exit_fits_region(exit_: Exit, region: Rect, grid_w: int, grid_d: int) -> Side | None:
    """The outer wall the exit sits on if it is on the footprint boundary and inside the
    connected region's wall segment; otherwise None."""
    side = exit_side(exit_, grid_w, grid_d)
    w = exit_.width_cells
    if side == Side.west:
        ok = exit_.x == 0 and region.x == 0 and region.y <= exit_.y and exit_.y + w <= region.y2
    elif side == Side.east:
        ok = (
            exit_.x == grid_w
            and region.x2 == grid_w
            and region.y <= exit_.y
            and exit_.y + w <= region.y2
        )
    elif side == Side.south:
        ok = exit_.y == 0 and region.y == 0 and region.x <= exit_.x and exit_.x + w <= region.x2
    elif side == Side.north:
        ok = (
            exit_.y == grid_d
            and region.y2 == grid_d
            and region.x <= exit_.x
            and exit_.x + w <= region.x2
        )
    else:
        return None
    return side if ok else None


def exit_center_m(exit_: Exit, side: Side, cell_size_m: float) -> tuple[float, float]:
    half = exit_.width_cells / 2
    if side in (Side.west, Side.east):
        return (exit_.x * cell_size_m, (exit_.y + half) * cell_size_m)
    return ((exit_.x + half) * cell_size_m, exit_.y * cell_size_m)
