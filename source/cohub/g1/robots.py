"""Unitree G1 articulation config for multi-robot scenes.

:func:`make_robot` builds a G1 at a pose; where each robot stands is set by
the task. Rotations are (x, y, z, w): the stock ``(0, 0, 0.7071, 0.7071)`` is
a +90 degree yaw facing world +Y, and ``(0, 0, -0.7071, 0.7071)`` faces -Y.
"""

import copy
import functools
import itertools
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from cohub.g1.geometry import PELVIS_SPAWN_Z  # noqa: F401

NATURAL_ARM_JOINT_POS = {
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
    "left_elbow_joint": -0.30,
    "right_elbow_joint": -0.30,
}
"""Standing arm pose that keeps both elbows off the torso [rad]."""


_SKIP_COLLIDERS = ("_hand_", "left_hand", "right_hand", "joints", "Looks")
"""Robot children that need no collider patch: the hands, and non-link scopes."""


def _has_enabled_collision_shape(link, Usd, UsdPhysics) -> bool:
    """Whether ``link`` already contains an enabled collision shape.

    An empty ``collisions`` scope or a shape with ``collisionEnabled=false``
    does not count.
    """
    container_types = {"", "Xform", "Scope", "Material", "SkelRoot", "Skeleton"}
    for candidate in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
        if candidate.GetTypeName() in container_types:
            continue
        schemas = {str(schema) for schema in candidate.GetAppliedSchemas()}
        try:
            has_collision_api = candidate.HasAPI(UsdPhysics.CollisionAPI)
        except (AttributeError, TypeError, RuntimeError):
            has_collision_api = False
        if not has_collision_api:
            has_collision_api = "PhysicsCollisionAPI" in schemas
        if not has_collision_api:
            continue
        collision_enabled = candidate.GetAttribute("physics:collisionEnabled")
        if collision_enabled.IsValid() and collision_enabled.HasAuthoredValue():
            if collision_enabled.Get() is False:
                continue
        return True
    return False


def _add_missing_colliders(prim_path_expr: str) -> int:
    """Give the G1's links that have no collider a convex hull, at spawn. Returns how many.

    Their ``visuals`` scopes are de-instanced first: USD cannot author through an instance proxy.
    """
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.utils.prims import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    patched = 0
    for robot_path in find_matching_prim_paths(prim_path_expr):
        robot = stage.GetPrimAtPath(robot_path)
        for link in robot.GetChildren():
            name = link.GetName()
            if any(s in name for s in _SKIP_COLLIDERS):
                continue
            if _has_enabled_collision_shape(link, Usd, UsdPhysics):
                continue
            for scope in link.GetChildren():
                if scope.IsInstanceable():
                    scope.SetInstanceable(False)
            meshes = [
                p
                for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
                if p.GetTypeName() == "Mesh" and not p.IsInstanceProxy()
            ]
            if not meshes:
                continue
            for mesh in meshes:
                UsdPhysics.CollisionAPI.Apply(mesh)
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(
                    UsdPhysics.Tokens.convexHull
                )
            patched += 1
    return patched


SELF_COLLISION_CHAIN_RADIUS = 2
"""Link pairs within this many joints of each other do not self-collide."""

_HAND_ROOTS = ("left_hand", "right_hand")
"""Prefixes of the Dex3 hand links. Pairs within one hand do not self-collide."""


def _self_collision_enabled() -> bool:
    """Whether to spawn robots that collide with themselves.

    On by default; ``COHUB_SELF_COLLISION=0`` turns it off.
    """
    return os.environ.get("COHUB_SELF_COLLISION", "1") not in ("0", "false", "False")


def _link_parents(robot, Usd, UsdPhysics) -> dict:
    """Map each link name to its parent's, read off the asset's joints."""
    parents = {}
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent = joint.GetBody0Rel().GetTargets()
        child = joint.GetBody1Rel().GetTargets()
        if not parent or not child:
            continue
        parents[child[0].name] = parent[0].name
    return parents


def _chain_distance(a: str, b: str, parents: dict) -> int:
    """Joints between links ``a`` and ``b``; a large number if unconnected."""

    def ancestors(name):
        depths, cur, d = {name: 0}, name, 0
        seen = {name}
        while cur in parents:
            cur = parents[cur]
            if cur in seen:
                break
            seen.add(cur)
            d += 1
            depths[cur] = d
        return depths

    up_a, up_b = ancestors(a), ancestors(b)
    shared = set(up_a) & set(up_b)
    return min(up_a[c] + up_b[c] for c in shared) if shared else 1 << 16


def _same_hand(a: str, b: str) -> bool:
    """Whether both links are fingers or palm of the *same* Dex3 hand."""
    return any(a.startswith(root) and b.startswith(root) for root in _HAND_ROOTS)


def _filter_self_collision_pairs(prim_path_expr: str) -> int:
    """Exclude the link pairs that overlap by construction, at spawn. Returns how many.

    A pair is excluded when its links are within :data:`SELF_COLLISION_CHAIN_RADIUS`
    joints of each other or belong to the same hand.
    """
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.utils.prims import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    filtered = 0
    for robot_path in find_matching_prim_paths(prim_path_expr):
        robot = stage.GetPrimAtPath(robot_path)
        parents = _link_parents(robot, Usd, UsdPhysics)
        bodies = {
            prim.GetName(): prim
            for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and _has_enabled_collision_shape(prim, Usd, UsdPhysics)
        }
        for a, b in itertools.combinations(sorted(bodies), 2):
            distance = _chain_distance(a, b, parents)
            # PhysX already excludes parent/child pairs.
            if distance < 2:
                continue
            if distance > SELF_COLLISION_CHAIN_RADIUS and not _same_hand(a, b):
                continue
            api = UsdPhysics.FilteredPairsAPI.Apply(bodies[a])
            api.CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
            filtered += 1
    return filtered


DEX3_PAD_FRICTION = (1.4, 1.2)
"""Static and dynamic friction of the Dex3's finger pads."""


def _bind_grip_material(
    prim_path_expr: str, friction: tuple[float, float], combine_mode: str | None = None
) -> int:
    """Give the hand's collision shapes a physics material, at spawn. Returns how many scopes.

    ``combine_mode`` is PhysX's friction combine mode (``None`` leaves it unset).
    """
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    from isaaclab.sim.utils.prims import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    bound = 0
    for robot_path in find_matching_prim_paths(prim_path_expr):
        material_path = f"{robot_path}/GripMaterial"
        material = UsdPhysics.MaterialAPI.Apply(
            UsdShade.Material.Define(stage, material_path).GetPrim()
        )
        material.CreateStaticFrictionAttr(friction[0])
        material.CreateDynamicFrictionAttr(friction[1])
        material.CreateRestitutionAttr(0.0)
        if combine_mode is not None:
            PhysxSchema.PhysxMaterialAPI.Apply(
                stage.GetPrimAtPath(material_path)
            ).CreateFrictionCombineModeAttr(combine_mode)
        for root in _HAND_ROOTS:
            hand = stage.GetPrimAtPath(f"{robot_path}/{root}")
            if not hand.IsValid():
                continue
            # Bind on the hand scope: its shapes are instance proxies and
            # cannot be bound directly.
            UsdShade.MaterialBindingAPI.Apply(hand).Bind(
                UsdShade.Material.Get(stage, material_path),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            bound += 1
    return bound


def with_grip_material(
    cfg: ArticulationCfg,
    friction: tuple[float, float] = DEX3_PAD_FRICTION,
    combine_mode: str | None = None,
):
    """Wrap a robot cfg's spawner so its hands spawn with a grip material."""
    original = cfg.spawn.func

    @functools.wraps(original)
    def wrapper(prim_path, spawn_cfg, *args, **kwargs):
        prim = original(prim_path, spawn_cfg, *args, **kwargs)
        _bind_grip_material(prim_path, friction, combine_mode)
        return prim

    cfg.spawn.func = wrapper
    return cfg


# Robot appearance

G1_SKIN_LIGHT = {
    "diffuse_color_constant": (0.70, 0.70, 0.71),
    "diffuse_tint": (0.79999995, 0.79999995, 0.76),
    "metallic_constant": 1.0,
    "metallic_texture_influence": 0.14999999105930328,
    "reflection_roughness_constant": 0.6499999761581421,
    "reflection_roughness_texture_influence": 0.0,
    "specular_level": 0.26999998092651367,
}
"""OmniPBR constants for the G1's light grey shell.

Taken from the Unitree asset published in
`SIMPLE <https://github.com/physical-superintelligence-lab/SIMPLE>`_
(``robots/g1/g1_29dof_wholebody_dex3.usd``)."""

G1_SKIN_DARK = {
    "diffuse_color_constant": (0.030888021, 0.029695433, 0.029695433),
    "diffuse_tint": (0.00386101, 0.0036970288, 0.0036970288),
}
"""OmniPBR constants for the black parts, from the same asset."""

G1_SKIN_LOGO = {"diffuse_color_constant": (0.87258685, 0.85574156, 0.85574156)}
"""Colour of the chest logo, off-white."""

G1_DARK_SKIN_LINKS = frozenset({
    "head_link",
    "waist_yaw_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
})
"""Link names that wear :data:`G1_SKIN_DARK`, besides the two hands."""

_SKIN_LOOKS = {
    "DefaultMaterial": G1_SKIN_LIGHT,
    "DefaultMaterial_0": G1_SKIN_LOGO,
}
"""Stock materials to retint in place, and the constants to set."""

_DARK_MATERIAL = "CoHuBDarkSkin"
"""Name of the material this module adds under the robot's ``Looks`` scope."""

_DARK_SKIN_SCOPES = frozenset(_HAND_ROOTS) | G1_DARK_SKIN_LINKS
"""Every scope that must be bound to the dark material."""


def _set_inputs(shader, values: dict) -> None:
    """Author OmniPBR constants on a shader, creating the inputs it lacks."""
    from pxr import Sdf, UsdShade

    for name, value in values.items():
        kind = Sdf.ValueTypeNames.Float if isinstance(value, float) else Sdf.ValueTypeNames.Color3f
        UsdShade.Shader(shader).CreateInput(name, kind).Set(value)


def _define_dark_material(stage, looks_path: str):
    """An OmniPBR material for the black parts, under the robot's own Looks."""
    from pxr import Sdf, UsdShade

    material_path = f"{looks_path}/{_DARK_MATERIAL}"
    existing = stage.GetPrimAtPath(material_path)
    if existing.IsValid():
        return UsdShade.Material(existing)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.GetPrim().CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set(
        "sourceAsset"
    )
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    _set_inputs(shader, G1_SKIN_DARK)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def _has_mesh(prim, Usd) -> bool:
    """Whether anything under ``prim`` is renderable geometry."""
    return any(
        child.GetTypeName() == "Mesh"
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
    )


def _make_authorable(stage, path, Usd):
    """De-instance the instance that holds ``path`` and return the prim at ``path``.

    USD cannot author through an instance proxy.
    """
    node = stage.GetPrimAtPath(path)
    while node.IsValid() and node.IsInstanceProxy():
        node = node.GetParent()
    if node.IsValid() and node.IsInstanceable():
        node.SetInstanceable(False)
    return stage.GetPrimAtPath(path)


def _dark_targets(stage, robot) -> dict:
    """``{scope name: prim}`` to bind :data:`G1_SKIN_DARK` on, in binding order.

    For each name, the target is the shallowest prim of that name that has
    geometry under it.
    """
    from pxr import Usd

    targets: dict[str, object] = {}
    for root in _HAND_ROOTS:
        hand = stage.GetPrimAtPath(f"{robot.GetPath()}/{root}")
        if hand.IsValid() and _has_mesh(hand, Usd):
            targets[root] = hand
    # Collect paths before writing: de-instancing invalidates the traversal.
    found: dict[str, object] = {}
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        name = prim.GetName()
        if name not in G1_DARK_SKIN_LINKS or name in found:
            continue
        if not _has_mesh(prim, Usd):
            continue
        found[name] = (prim.GetPath(), prim.IsInstanceProxy())
    for name, (path, is_proxy) in found.items():
        prim = _make_authorable(stage, path, Usd) if is_proxy else stage.GetPrimAtPath(path)
        if prim.IsValid():
            targets[name] = prim
    return targets


def _skin_robot(stage, robot_path: str) -> int:
    """Repaint one spawned G1 in Unitree's own colours, in place.

    Retints the stock materials (:data:`_SKIN_LOOKS`) and binds one dark material
    over the black parts (:data:`_DARK_SKIN_SCOPES`). Raises ``RuntimeError`` if
    either is missing. Returns how many scopes were bound.
    """
    from pxr import UsdShade

    robot = stage.GetPrimAtPath(robot_path)
    if not robot.IsValid():
        raise RuntimeError(f"G1 skin: {robot_path} is not a prim on this stage")
    looks_path = f"{robot_path}/Looks"
    for name, values in _SKIN_LOOKS.items():
        material = stage.GetPrimAtPath(f"{looks_path}/{name}")
        shaders = (
            [child for child in material.GetChildren() if child.GetTypeName() == "Shader"]
            if material.IsValid()
            else []
        )
        if not shaders:
            raise RuntimeError(
                f"G1 skin: no shader at {looks_path}/{name} to retint. The light grey is "
                "authored onto the asset's own materials, so a G1 that does not have them "
                "spawns in Isaac's chrome."
            )
        for shader in shaders:
            _set_inputs(shader, values)
    dark = _define_dark_material(stage, looks_path)
    targets = _dark_targets(stage, robot)
    missing = sorted(_DARK_SKIN_SCOPES - set(targets))
    if missing:
        raise RuntimeError(
            f"G1 skin: no geometry under {robot_path} for {', '.join(missing)}. "
            "Those parts are black on the real robot; unbound they stay chrome."
        )
    for target in targets.values():
        UsdShade.MaterialBindingAPI.Apply(target).Bind(
            dark, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
    return len(targets)


def _bind_skin(prim_path_expr: str) -> int:
    """Skin every G1 matching ``prim_path_expr``, at spawn. Raises ``RuntimeError`` if none matches."""
    from isaaclab.sim.utils.prims import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    robot_paths = find_matching_prim_paths(prim_path_expr)
    if not robot_paths:
        raise RuntimeError(
            f"G1 skin: {prim_path_expr} matched no prim, so nothing was repainted."
        )
    return sum(_skin_robot(stage, path) for path in robot_paths)


def _spawn_with_full_colliders(original):
    """Wrap a USD spawner so the robot lands with every limb collidable."""

    @functools.wraps(original)
    def wrapper(prim_path, cfg, *args, **kwargs):
        prim = original(prim_path, cfg, *args, **kwargs)
        _add_missing_colliders(prim_path)
        if cfg.articulation_props is not None and cfg.articulation_props.enabled_self_collisions:
            _filter_self_collision_pairs(prim_path)
        _bind_skin(prim_path)
        return prim

    return wrapper


def make_robot(prim_name: str, pos, rot=None) -> ArticulationCfg:
    """A G1 articulation cfg named ``prim_name`` at ``pos`` [m] (use ``PELVIS_SPAWN_Z``).

    ``rot`` is (x, y, z, w); ``None`` keeps the stock +Y-facing orientation.
    """
    cfg = copy.deepcopy(G1_29DOF_CFG)
    cfg.prim_path = "{ENV_REGEX_NS}/" + prim_name
    usd_override = os.environ.get("COHUB_G1_USD")
    if usd_override:
        cfg.spawn.usd_path = usd_override
    cfg.spawn.activate_contact_sensors = True
    cfg.spawn.articulation_props.enabled_self_collisions = _self_collision_enabled()
    # Patch missing colliders and the skin as the robot spawns.
    cfg.spawn.func = _spawn_with_full_colliders(sim_utils.spawn_from_usd)
    cfg.init_state.pos = pos
    cfg.init_state.joint_pos.update(NATURAL_ARM_JOINT_POS)
    if rot is not None:
        cfg.init_state.rot = rot
    return cfg
