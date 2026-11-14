"""2D navigation planning over the Ego-State.

Floor-level ES entities (bottom below ``floor_obstacle_height``) become oriented rectangles on
an occupancy grid, inflated by the robot's circumscribed radius; A* finds an 8-connected path
that is shortened by line-of-sight checks. Docking manoeuvres near furniture use exact
rectangle-overlap checks of the robot footprint instead of the inflated grid.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from steward.math3d import yaw_from_quat
from steward.scene.schema import SceneState

FLOOR_LYING_M = 0.05  # an object whose estimated base is lower than this lies on the floor


@dataclass
class Box2D:
    id: str
    cx: float
    cy: float
    yaw: float
    hx: float
    hy: float

    def corners(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        r = np.array([[c, -s], [s, c]])
        local = np.array([[self.hx, self.hy], [-self.hx, self.hy], [-self.hx, -self.hy], [self.hx, -self.hy]])
        return local @ r.T + np.array([self.cx, self.cy])

    def contains(self, x: float, y: float, inflate: float = 0.0) -> bool:
        dx, dy = x - self.cx, y - self.cy
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= self.hx + inflate and abs(ly) <= self.hy + inflate

    def distance(self, x: float, y: float) -> float:
        dx, dy = x - self.cx, y - self.cy
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        lx, ly = abs(c * dx - s * dy) - self.hx, abs(s * dx + c * dy) - self.hy
        return math.hypot(max(lx, 0.0), max(ly, 0.0)) - (0.0 if lx > 0 or ly > 0 else min(-lx, -ly))


def entity_box(ent) -> tuple[Box2D, float, float]:
    """Footprint box of a single-link ES entity plus its bottom and top heights."""
    geom = ent.rigid.links[0].geometry["collision"][0]
    sx, sy, sz = geom["size"]
    p = ent.pose.value["p"]
    box = Box2D(ent.id, p[0], p[1], yaw_from_quat(ent.pose.value["q"]), sx / 2, sy / 2)
    return box, p[2] - sz / 2, p[2] + sz / 2


def floor_obstacles(es: SceneState, exclude: set[str], floor_obstacle_height: float = 0.35,
                    min_existence: float = 0.3, ground_clearance: float = 0.0) -> list[Box2D]:
    """Footprints the base must keep out of: what reaches below ``floor_obstacle_height``, except a
    flat thing lying on the floor no thicker than the base's ``ground_clearance``, which it drives
    over (judged by its thickness: its height estimate is the noisier one)."""
    out = []
    for ent in es.entities:
        if ent.id == es.self_id or ent.id in exclude or ent.existence < min_existence:
            continue
        box, bottom, top = entity_box(ent)
        flat_on_floor = top - bottom <= ground_clearance and bottom < FLOOR_LYING_M
        if bottom < floor_obstacle_height and not flat_on_floor:
            out.append(box)
    return out


def front_box(box: Box2D, room_centre: tuple[float, float] | None) -> Box2D:
    """The same box re-expressed so that its +x axis points out of the face that looks into the room
    (towards the room centre), with ``hx`` the half-extent along that axis. Furniture stands with its
    back to a wall, so this is its front whatever frame the box was estimated in."""
    if room_centre is None:
        return box
    to_room = (room_centre[0] - box.cx, room_centre[1] - box.cy)
    best = None
    for k in range(4):
        yaw = box.yaw + k * math.pi / 2
        hx, hy = (box.hx, box.hy) if k % 2 == 0 else (box.hy, box.hx)
        score = math.cos(yaw) * to_room[0] + math.sin(yaw) * to_room[1]
        if best is None or score > best[0]:
            best = (score, Box2D(box.id, box.cx, box.cy, (yaw + math.pi) % (2 * math.pi) - math.pi, hx, hy))
    return best[1]


def room_centre(boxes: list[Box2D]) -> tuple[float, float] | None:
    """Middle of everything known in the room (the extent of all boxes), or ``None`` when empty."""
    if not boxes:
        return None
    xs = [c for b in boxes for c in b.corners()[:, 0]]
    ys = [c for b in boxes for c in b.corners()[:, 1]]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


def protrusion(drawer: Box2D, furniture: list[Box2D]) -> float:
    """How far a drawer box sticks out past the front of the furniture box that holds it, along the
    drawer's +x axis (pass a :func:`front_box`); 0 when no furniture box contains the drawer's back end."""
    n = (math.cos(drawer.yaw), math.sin(drawer.yaw))
    front = drawer.cx * n[0] + drawer.cy * n[1] + drawer.hx
    back = (drawer.cx - n[0] * (drawer.hx - 0.03), drawer.cy - n[1] * (drawer.hx - 0.03))
    best = None
    for box in furniture:
        if box.id == drawer.id or not box.contains(back[0], back[1], inflate=0.05):
            continue
        reach = max((c[0] * n[0] + c[1] * n[1]) for c in box.corners())
        best = front - reach if best is None else min(best, front - reach)
    return max(0.0, best) if best is not None else 0.0


def rects_overlap(a: Box2D, b: Box2D) -> bool:
    """Separating-axis test for two oriented rectangles."""
    pa, pb = a.corners(), b.corners()
    for poly in (pa, pb):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            proj_a, proj_b = pa @ axis, pb @ axis
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True


def rectangle_separation(a: Box2D, b: Box2D) -> float:
    """Signed SAT gap: negative is penetration; positive is a separating axis."""
    pa, pb = a.corners(), b.corners()
    gaps = []
    for yaw in (a.yaw, b.yaw):
        for axis in ((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))):
            aa, bb = pa @ axis, pb @ axis
            gaps.append(max(float(bb.min()-aa.max()), float(aa.min()-bb.max())))
    return max(gaps)


def escape_prefix(obstacles, start, yaw, footprint, *, carry_box=None,
                  turn_radius, margin, max_distance, max_overlap=.05):
    """Fixed-heading translation out of shallow estimated overlap, never rotate in it.

    Every initially intersecting pair must monotonically separate; other pairs
    must remain separate. Search forward/backward, also usable by wheeled bases.
    The returned endpoint has clearance for the full turning envelope.
    """
    def bodies(xy):
        out = [Box2D('base', *xy, yaw, *footprint)]
        if carry_box is not None:
            out.append(carry_box(xy))
        return out
    initial = [rectangle_separation(a, b) for a in bodies(start) for b in obstacles]
    if any(d < -max_overlap for d in initial):
        return None
    for sign in (-1, 1):
        previous = initial
        for distance in np.arange(.01, max_distance+.001, .01):
            xy = (start[0]+sign*distance*math.cos(yaw), start[1]+sign*distance*math.sin(yaw))
            gaps = [rectangle_separation(a, b) for a in bodies(xy) for b in obstacles]
            if any((old < 0 and new < old-1e-8) or (old >= 0 and new <= 0)
                   for old, new in zip(previous, gaps)):
                break
            # No sliding sideways deeper along an existing overlap plateau.
            if any(first < 0 and new <= first+1e-8 for first, new in zip(initial, gaps)):
                break
            if all(b.distance(*xy) >= turn_radius+margin for b in obstacles):
                return [float(xy[0]), float(xy[1]), float(yaw)]
            previous = gaps
    return None


@dataclass
class Grid:
    x0: float
    y0: float
    res: float
    occ: np.ndarray

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round((x - self.x0) / self.res)), int(round((y - self.y0) / self.res))

    def point(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + i * self.res, self.y0 + j * self.res

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]

    def free(self, i: int, j: int) -> bool:
        return self.inside(i, j) and not self.occ[i, j]


def build_grid(obstacles: list[Box2D], inflate: float, around: list[tuple[float, float]], res: float = 0.05,
               margin: float = 1.0) -> Grid:
    xs = [p[0] for p in around] + [c for b in obstacles for c in b.corners()[:, 0]]
    ys = [p[1] for p in around] + [c for b in obstacles for c in b.corners()[:, 1]]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    nx, ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    gx = x0 + np.arange(nx) * res
    gy = y0 + np.arange(ny) * res
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    occ = np.zeros((nx, ny), dtype=bool)
    for b in obstacles:
        c, s = math.cos(-b.yaw), math.sin(-b.yaw)
        dx, dy = xx - b.cx, yy - b.cy
        lx, ly = c * dx - s * dy, s * dx + c * dy
        occ |= (np.abs(lx) <= b.hx + inflate) & (np.abs(ly) <= b.hy + inflate)
    return Grid(x0, y0, res, occ)


def astar(grid: Grid, start: tuple[float, float], goal: tuple[float, float], start_clear: float = 0.2) -> list | None:
    si, sj = grid.cell(*start)
    gi, gj = grid.cell(*goal)
    if not grid.inside(si, sj) or not grid.inside(gi, gj) or grid.occ[gi, gj]:
        return None
    r = int(start_clear / grid.res)
    occ = grid.occ.copy()
    occ[max(0, si - r) : si + r + 1, max(0, sj - r) : sj + r + 1] = False
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
    open_heap = [(0.0, 0.0, si, sj)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {(si, sj): 0.0}
    while open_heap:
        _, g, i, j = heapq.heappop(open_heap)
        if (i, j) == (gi, gj):
            path = [(i, j)]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            pts = [grid.point(a, b) for a, b in path]
            pts[0], pts[-1] = tuple(start), tuple(goal)
            return _shortcut(Grid(grid.x0, grid.y0, grid.res, occ), pts)
        if g > cost.get((i, j), math.inf):
            continue
        for di, dj, w in moves:
            ni, nj = i + di, j + dj
            if not (0 <= ni < occ.shape[0] and 0 <= nj < occ.shape[1]) or occ[ni, nj]:
                continue
            ng = g + w
            if ng < cost.get((ni, nj), math.inf):
                cost[(ni, nj)] = ng
                came[(ni, nj)] = (i, j)
                heapq.heappush(open_heap, (ng + math.hypot(gi - ni, gj - nj), ng, ni, nj))
    return None


def _line_free(grid: Grid, a: tuple[float, float], b: tuple[float, float]) -> bool:
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (grid.res / 2)))
    for t in np.linspace(0.0, 1.0, n):
        i, j = grid.cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not grid.free(i, j):
            return False
    return True


def _shortcut(grid: Grid, pts: list) -> list:
    out = [pts[0]]
    k = 0
    while k < len(pts) - 1:
        nxt = len(pts) - 1
        while nxt > k + 1 and not _line_free(grid, pts[k], pts[nxt]):
            nxt -= 1
        out.append(pts[nxt])
        k = nxt
    return out


def path_length(path: list) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:], strict=False))


def path_clearance(path: list, obstacles: list[Box2D], step: float = 0.05) -> float:
    best = math.inf
    for a, b in zip(path, path[1:], strict=False):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for t in np.linspace(0.0, 1.0, n):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            for box in obstacles:
                best = min(best, box.distance(x, y))
    return best
