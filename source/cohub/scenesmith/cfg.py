"""Place a downloaded SceneSmith scene into a task's scene cfg."""

from __future__ import annotations

import functools
import math

from dataclasses import dataclass

from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg

SETUP_HINT = "python scripts/setup_assets.py rooms"


def yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    """A yaw about Z in (x, y, z, w), the order the spawner takes."""
    half = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def scene_data_dir(scene_id: str) -> Path:
    """Where ``scripts/setup_assets.py rooms`` puts a scene, given its id (``Room/scene_085``)."""
    return Path(__file__).resolve().parents[3] / "external/scenesmith_data/scenes" / scene_id.split("/", 1)[1]


@dataclass(frozen=True)
class Move:
    """Where some of a room's own furniture goes, in task-frame metres.

    ``names`` are whole object names. ``turn_about`` turns the group a half turn
    about that (x, y) before shifting it by ``dx``, ``dy``.
    """

    names: tuple[str, ...]
    dx: float = 0.0
    dy: float = 0.0
    turn_about: tuple[float, float] | None = None


@dataclass(frozen=True)
class Widen:
    """Cut one of the room's own openings wider, in task metres.

    ``names`` are the wall prims to cut (both rooms' walls for a partition).
    ``across`` are the opening's current edges along ``axis``, ``to`` where they go.
    """

    names: tuple[str, ...]
    across: tuple[float, float]
    to: tuple[float, float]
    axis: str = "y"
    tol: float = 0.002


def _move_index(moves) -> dict:
    """Map each name to its :class:`Move`. Raises if a name is in two groups."""
    index: dict = {}
    for move in moves:
        for name in move.names:
            if name in index:
                raise ValueError(f"{name!r} is in two Move groups; it can only go one place")
            index[name] = move
    return index

def _apply_moves(stage, scope: str, moves, yaw_deg: float,
                 offset: tuple[float, float]) -> None:
    """Re-pose or deactivate the named children, in task metres."""
    from pxr import Gf, UsdGeom

    index = _move_index(moves)
    rad = math.radians(yaw_deg)
    cos, sin = math.cos(rad), math.sin(rad)

    def to_task(x: float, y: float) -> tuple[float, float]:
        return (cos * x - sin * y + offset[0], sin * x + cos * y + offset[1])

    def to_room(x: float, y: float) -> tuple[float, float]:
        x, y = x - offset[0], y - offset[1]
        return (cos * x + sin * y, -sin * x + cos * y)

    seen = set()
    for child in stage.GetPrimAtPath(scope).GetChildren():
        move = index.get(child.GetName())
        if move is None:
            continue
        seen.add(child.GetName())
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(child).GetOrderedXformOps()}
        translate = ops.get("xformOp:translate")
        if translate is None:
            continue
        here = translate.Get()
        x, y = to_task(here[0], here[1])
        if move.turn_about is not None:
            px, py = move.turn_about
            x, y = 2.0 * px - x, 2.0 * py - y
            orient = ops.get("xformOp:orient")
            if orient is not None:
                # A half turn about z is the same in the task and room frames.
                quat = orient.Get()
                axis = quat.GetImaginary()
                half = type(quat)(0.0, type(axis)(0.0, 0.0, 1.0))
                orient.Set(half * quat)
        x, y = x + move.dx, y + move.dy
        room_x, room_y = to_room(x, y)
        translate.Set(type(here)(room_x, room_y, here[2]))

    missing = sorted(set(index) - seen)
    if missing:
        raise ValueError(f"Move names not found in this scene: {', '.join(missing)}")

def _widen(stage, room_path: str, widen) -> None:
    """Move an opening's jambs, in the visual mesh and in the colliders beside it.

    Works in the task frame by composing each prim's transform up to the
    room's parent.
    """
    from pxr import Gf, Usd, UsdGeom

    now = Usd.TimeCode.Default()
    room = stage.GetPrimAtPath(room_path)
    env = UsdGeom.Xformable(room.GetParent()).ComputeLocalToWorldTransform(now)
    into_task = env.GetInverse()
    lane = {"x": 0, "y": 1}[widen.axis]
    edges = tuple(widen.across)
    targets = tuple(widen.to)

    def moved(value: float) -> float | None:
        for edge, target in zip(edges, targets):
            if abs(value - edge) <= widen.tol:
                return target
        return None

    found = set()
    for prim in Usd.PrimRange(room):
        if prim.GetName() not in widen.names:
            continue
        local = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(now) * into_task
        inverse = local.GetInverse()
        mesh = UsdGeom.Mesh(prim)
        if mesh:
            points = list(mesh.GetPointsAttr().Get())
            touched = 0
            for index, point in enumerate(points):
                here = local.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                target = moved(here[lane])
                if target is None:
                    continue
                here[lane] = target
                back = inverse.Transform(here)
                points[index] = type(point)(back[0], back[1], back[2])
                touched += 1
            if touched:
                mesh.GetPointsAttr().Set(points)
                found.add(prim.GetName())
            continue
        cube = UsdGeom.Cube(prim)
        if not cube:
            continue
        # A solid wall segment: resize its span along the lane.
        half = cube.GetSizeAttr().Get() / 2.0
        corners = [
            local.Transform(Gf.Vec3d(sx * half, sy * half, sz * half))[lane]
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
        ]
        low, high = min(corners), max(corners)
        new_low, new_high = moved(low), moved(high)
        if new_low is None and new_high is None:
            continue
        low, high = (new_low if new_low is not None else low), (new_high if new_high is not None else high)
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        translate, scale = ops.get("xformOp:translate"), ops.get("xformOp:scale")
        if translate is None or scale is None:
            continue
        # Which of the prim's own axes runs along the lane. Rows are
        # normalised because ``xformOp:translate`` is not scaled.
        rows = [local.GetRow3(axis) for axis in range(3)]
        column = [(row / (row.GetLength() or 1.0))[lane] for row in rows]
        own = max(range(3), key=lambda axis: abs(column[axis]))
        span_now = max(corners) - min(corners)
        factor = (high - low) / span_now if span_now else 1.0
        centre_now = (max(corners) + min(corners)) / 2.0
        centre_new = (high + low) / 2.0
        values = list(scale.Get())
        values[own] = values[own] * factor
        scale.Set(type(scale.Get())(*values))
        shift = (centre_new - centre_now) / column[own]
        here = translate.Get()
        moved_local = [here[0], here[1], here[2]]
        moved_local[own] += shift
        translate.Set(type(here)(*moved_local))
        found.add(prim.GetName())

    missing = sorted(set(widen.names) - found)
    if missing:
        raise ValueError(
            f"Widen touched nothing on: {', '.join(missing)}. Either the name is not in "
            f"this scene, or `across` is not where that wall's opening is."
        )

def _selects(entry: str, name: str) -> bool:
    """Whether one ``hide``/``lower`` entry names this object.

    A plain entry matches as a substring. A trailing ``$`` anchors it to the
    end of the name.
    """
    return name.endswith(entry[:-1]) if entry.endswith("$") else entry in name


def _lower_prim(prim, by: float) -> bool:
    """Drop one top-level scene object ``by`` metres, in place.

    Edits the object's existing ``xformOp:translate``.
    """
    from pxr import Gf, UsdGeom

    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() != UsdGeom.XformOp.TypeTranslate:
            continue
        value = op.Get()
        op.Set(Gf.Vec3d(value[0], value[1], value[2] - by))
        return True
    return False


def _guarded_spawn(
    func,
    usd: Path,
    hide: tuple[str, ...],
    lower: tuple[str, ...],
    lower_by: float,
    scene_id: str | None,
    rearrange=(),
    yaw_deg: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
):
    """Validate at spawn time, and move what the task needs moved.

    Raises ``FileNotFoundError`` at spawn if the scene is not downloaded.
    ``hide`` and ``lower`` run first, then ``rearrange``.
    """

    @functools.wraps(func)
    def wrapper(prim_path, cfg, *args, **kwargs):
        if not usd.exists():
            what = f" ({scene_id})" if scene_id else ""
            raise FileNotFoundError(
                f"this task's room{what} is not downloaded yet:\n  {usd}\n"
                f"Get it with:\n  {SETUP_HINT}"
            )
        prim = func(prim_path, cfg, *args, **kwargs)
        if not (hide or (lower and lower_by) or rearrange):
            return prim

        import isaaclab.sim.utils.prims as prim_utils

        stage = prim_utils.get_current_stage()
        scope = f"{prim_path}/Geometry"
        lowered, missed = 0, set(lower)
        for child in stage.GetPrimAtPath(scope).GetChildren():
            name = child.GetName()
            if any(_selects(n, name) for n in hide):
                child.SetActive(False)
                continue
            hit = [n for n in lower if _selects(n, name)]
            if hit and lower_by and _lower_prim(child, lower_by):
                lowered += 1
                missed.difference_update(hit)
        if missed:
            # A name that matches nothing would be a silent no-move.
            raise ValueError(
                f"lower= matched no object in {scene_id or usd.name}: "
                f"{sorted(missed)}\nCheck the names against the children "
                f"of the scene's Geometry prim."
            )
        if lowered:
            print(f"[scenesmith] lowered {lowered} object(s) by {lower_by:.3f} m")
        placed = tuple(item for item in rearrange if not isinstance(item, Widen))
        if placed:
            _apply_moves(stage, scope, placed, yaw_deg, offset)
        for item in rearrange:
            if isinstance(item, Widen):
                _widen(stage, prim_path, item)
        return prim

    return wrapper


def scenery_cfg_at(
    scene_dir: str | Path,
    origin_at: tuple[float, float],
    yaw_deg: float = 0.0,
    name: str = "Room",
    hide: tuple[str, ...] = (),
    lower: tuple[str, ...] = (),
    lower_by: float = 0.0,
    scene_id: str | None = None,
    rearrange: tuple = (),
) -> AssetBaseCfg:
    """A prepared scene, turned and shifted so it sits around the task.

    ``origin_at`` is the scene (x, y) that lands on the task's origin; ``yaw_deg``
    turns the room about it (right angles only). ``hide`` and ``lower`` take
    substrings of top-level object names (a trailing ``$`` anchors one); name what
    rests on them too. ``rearrange`` takes :class:`Move` and :class:`Widen` records.
    """
    # Absolute: USD yields an empty prim for an unresolved relative path.
    usd = (Path(scene_dir) / "scene_static.usda").resolve()
    rad = math.radians(yaw_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    ox, oy = origin_at
    # Translate by the rotated anchor, negated.
    pos = (-(cos * ox - sin * oy), -(sin * ox + cos * oy), 0.0)

    spawn = sim_utils.UsdFileCfg(usd_path=str(usd))
    spawn.func = _guarded_spawn(
        spawn.func, usd, tuple(hide), tuple(lower), float(lower_by), scene_id,
        tuple(rearrange), yaw_deg, (pos[0], pos[1]),
    )
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=yaw_quat(yaw_deg)),
    )
