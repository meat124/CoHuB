"""Spawners and world lighting shared by the tasks."""

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

# Isaac Lab initial-state rotations are (x, y, z, w).
IDENTITY_ROT = (0.0, 0.0, 0.0, 1.0)
YAW_90 = (0.0, 0.0, 0.70710678, 0.70710678)
YAW_180 = (0.0, 0.0, 1.0, 0.0)
YAW_270 = (0.0, 0.0, -0.70710678, 0.70710678)

# Green goal paint
GOAL_PAINT_COLOUR = (0.1, 0.75, 0.25)
GOAL_PAINT_EMISSIVE = (0.05, 0.35, 0.1)
GOAL_PAINT_THICKNESS = 0.003

# Offsets that keep coplanar surfaces from z-fighting

COPLANAR_LIFT = 0.002
"""Lift of a visual-only floor slab over the floor under it [m]."""

DECAL_LIFT = 0.001
"""Lift of a painted marker over the face it is painted on [m]."""

# Fill colours for the openings of a downloaded room
DOOR_FILL_COLOUR = (0.55, 0.47, 0.38)
"""Light wood, for a filled doorway."""

WINDOW_FILL_COLOUR = (0.80, 0.84, 0.86)
"""Pale blue-grey, for a filled window."""


def box_cfg(
    prim_name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    collision: bool = True,
    visible: bool = True,
    emissive: tuple[float, float, float] | None = None,
    movable: bool = False,
    roughness: float = 0.72,
) -> AssetBaseCfg | RigidObjectCfg:
    """One kinematic box, positioned by its centre.

    ``emissive`` marks a painted surface. ``movable`` returns a
    ``RigidObjectCfg``, which an event term can move at reset.
    """
    surface = {"diffuse_color": color, "roughness": roughness}
    if emissive is not None:
        surface["emissive_color"] = emissive
    spawn_kwargs = {
        "size": size,
        "visible": visible,
        "visual_material": sim_utils.PreviewSurfaceCfg(**surface),
    }
    if collision:
        spawn_kwargs.update(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    elif movable:
        # A RigidObject needs a rigid body even without a collider.
        spawn_kwargs["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    cls = RigidObjectCfg if movable else AssetBaseCfg
    return cls(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        init_state=cls.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(**spawn_kwargs),
    )


CEILING_THICKNESS = 0.05
"""Ceiling slab thickness [m]."""

_CEILING_USD = Path(__file__).parent / "data" / "ceiling.usda"


def ceiling_cfg(
    size: tuple[float, float],
    centre: tuple[float, float],
    top_z: float,
    prim_name: str = "Ceiling",
) -> AssetBaseCfg:
    """A ceiling over a downloaded room, resting on its wall tops.

    ``size`` and ``centre`` are the room's footprint in the task frame and
    ``top_z`` is the height of its walls. The slab sits above ``top_z``. Its
    USD casts no shadow, so the dome light still lights the room.
    """
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(centre[0], centre[1], top_z + CEILING_THICKNESS / 2)
        ),
        spawn=UsdFileCfg(
            usd_path=str(_CEILING_USD),
            scale=(size[0], size[1], CEILING_THICKNESS),
        ),
    )


# Physics settings for props a robot holds or pushes

MANIPULAND_CONTACT_OFFSET = 0.006
"""Default PhysX contact offset for a held or pushed body [m].
Pass a smaller value for a small prop."""


def manipuland_rigid_props(**overrides) -> sim_utils.RigidBodyPropertiesCfg:
    """Rigid-body settings for a prop a robot manipulates.

    ``**overrides`` replaces any of the fields.
    """
    return sim_utils.RigidBodyPropertiesCfg(
        **{
            "max_depenetration_velocity": 0.25,
            "solver_position_iteration_count": 16,
            "solver_velocity_iteration_count": 2,
            "sleep_threshold": 0.0,
            "disable_gravity": False,
            "enable_gyroscopic_forces": True,
            **overrides,
        }
    )


def manipuland_collision_props(
    contact_offset: float = MANIPULAND_CONTACT_OFFSET,
) -> sim_utils.CollisionPropertiesCfg:
    """Collision settings for a prop a robot manipulates.

    ``rest_offset`` is 0 so the prop rests on the surface. See
    :data:`MANIPULAND_CONTACT_OFFSET`.
    """
    return sim_utils.CollisionPropertiesCfg(contact_offset=contact_offset, rest_offset=0.0)


def polyhaven_cfg(
    root: Path,
    prim_name: str,
    asset_dir: str,
    pos: tuple[float, float, float],
    *,
    filename: str | None = None,
    rot: tuple[float, float, float, float] = IDENTITY_ROT,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> AssetBaseCfg:
    """Spawn a vendored Poly Haven CC0 model from ``root``, visual only.

    ``filename`` defaults to ``<asset_dir>_1k.usdc``.
    """
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=UsdFileCfg(
            usd_path=str(root / asset_dir / (filename or f"{asset_dir}_1k.usdc")),
            scale=scale,
        ),
    )


_GRID_USD = os.environ.get("COHUB_GRID_USD", GroundPlaneCfg().usd_path)

GROUND_CFG = AssetBaseCfg(
    prim_path="/World/GroundPlane",
    spawn=GroundPlaneCfg(usd_path=_GRID_USD),
)

LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/light",
    # Lights the room without being drawn.
    spawn=sim_utils.DomeLightCfg(
        color=(0.75, 0.75, 0.75), intensity=3000.0, visible_in_primary_ray=False
    ),
)


# Additions for a task in a downloaded SceneSmith room

def hidden_grid_cfg() -> AssetBaseCfg:
    """The platform ground plane with its grid not drawn.

    The collider stays: the robot spawn height and the XR anchor are
    measured against this plane at z=0.
    """
    return GROUND_CFG.replace(spawn=GROUND_CFG.spawn.replace(visible=False))


def carried_table_cfg(usd_path: Path, pos: tuple[float, float, float], mass: float) -> RigidObjectCfg:
    """A table the robots carry: a baked USD as the rigid body ``{ENV_REGEX_NS}/Table``."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.25,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                # A sleeping body reports no contact to the grasp term.
                sleep_threshold=0.0,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.006, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        ),
    )


DOOR_LEAF_COLOUR = (0.45, 0.33, 0.24)
DOOR_LEAF_THICKNESS = 0.08
DOOR_LEAF_REVEAL = 0.005
"""A closed door leaf: colour, thickness [m] and gap to the opening on each edge [m]."""


def doorway_panel_cfg(
    name: str,
    opening: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    origin_at: tuple[float, float],
    yaw_deg: float = 0.0,
    *,
    color: tuple[float, float, float] = DOOR_LEAF_COLOUR,
):
    """A closed door leaf filling a doorway of a downloaded room. Right-angle yaws only.

    ``opening`` is the hole in scene coordinates, ``((x0, x1), (y0, y1), (z0, z1))``;
    ``origin_at`` and ``yaw_deg`` are the room's own :func:`scenery_cfg_at` values.
    """
    quarter = yaw_deg / 90.0
    if abs(quarter - round(quarter)) > 1e-9:
        raise ValueError(f"doorway_panel_cfg: {name}: yaw {yaw_deg} is not a right angle")
    (x0, x1), (y0, y1), (z0, z1) = opening
    extent = [x1 - x0, y1 - y0, z1 - z0]
    thin = 0 if extent[0] < extent[1] else 1  # the wall's thickness axis
    size = [extent[0] - 2 * DOOR_LEAF_REVEAL, extent[1] - 2 * DOOR_LEAF_REVEAL, extent[2] - 2 * DOOR_LEAF_REVEAL]
    size[thin] = DOOR_LEAF_THICKNESS
    # Scene coordinates to the task frame.
    rad = math.radians(yaw_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    dx, dy = (x0 + x1) / 2.0 - origin_at[0], (y0 + y1) / 2.0 - origin_at[1]
    pos = (cos * dx - sin * dy, sin * dx + cos * dy, (z0 + z1) / 2.0)
    if round(quarter) % 2:
        size[0], size[1] = size[1], size[0]
    return box_cfg(name, size=tuple(size), pos=pos, color=color)
