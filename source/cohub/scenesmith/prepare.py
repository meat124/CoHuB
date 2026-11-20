"""Prepare a downloaded SceneSmith scene: pin it, give its materials surface values, retexture its windows."""

from __future__ import annotations

import glob
import os
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics, UsdShade


# --- Pinning the scene ---
# A published scene arrives with every object a free rigid body. The overlay sublayers the scene and
# sets kinematicEnabled on what to freeze; articulated furniture is frozen whole.

def scene_usd(scene_dir: str | Path) -> Path:
    """The scene's USD entry point, as published under ``mujoco/usd/``."""
    scene_dir = Path(scene_dir)
    hits = sorted(glob.glob(str(scene_dir / "mujoco" / "usd" / "scene_*.usda")))
    if not hits:
        raise FileNotFoundError(
            f"no mujoco/usd/scene_*.usda under {scene_dir}. Point this at an "
            "unpacked scene directory from nepfaff/scenesmith-example-scenes."
        )
    return Path(hits[0])


def prepare(
    scene_dir: str | Path,
    out: str | Path | None = None,
    dynamic: list[str] | None = None,
) -> tuple[Path, int, int]:
    """Write the kinematic overlay beside the scene (``<scene_dir>/scene_static.usda``).

    ``dynamic`` are substrings of object names to leave simulated. Returns the
    overlay path and the counts of bodies pinned, left free, and joints deactivated.
    """
    scene_dir = Path(scene_dir)
    src = scene_usd(scene_dir).resolve()
    dst = Path(out).resolve() if out else (scene_dir / "scene_static.usda").resolve()
    dynamic = dynamic or []

    if dst.exists():
        dst.unlink()
    src_layer = Sdf.Layer.FindOrOpen(str(src))
    if src_layer is None:
        raise FileNotFoundError(src)

    layer = Sdf.Layer.CreateNew(str(dst))
    # Relative when possible, so the overlay travels with the scene.
    try:
        layer.subLayerPaths.append(os.path.relpath(src, dst.parent))
    except ValueError:
        layer.subLayerPaths.append(str(src))
    # A sublayer does not carry the source's defaultPrim.
    layer.defaultPrim = src_layer.defaultPrim

    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(Usd.EditTarget(layer))

    articulated = [
        p.GetPath().pathString
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]

    pinned = free = joints = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            path = prim.GetPath().pathString
            if not any(name in path for name in dynamic):
                prim.SetActive(False)
                joints += 1
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = prim.GetPath().pathString
        if any(name in path for name in dynamic):
            free += 1
            continue
        attr = prim.GetAttribute("physics:kinematicEnabled")
        if not attr:
            attr = prim.CreateAttribute(
                "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool
            )
        attr.Set(True)
        pinned += 1

    layer.Save()
    return dst, pinned, free, joints


# --- Surface values ---
# SceneSmith exports every material with roughness 0.5 and metallic 0. Materials are matched by name
# substring; the first match in SURFACES wins.

SURFACES: tuple[tuple[tuple[str, ...], float, float, str], ...] = (
    # (name substrings, roughness, metallic, what it is)
    (("window",), 0.05, 0.0, "glass"),
    (("stanchion_post",), 0.30, 1.0, "brushed metal rope post"),
    (("service_desk_bell",), 0.20, 1.0, "polished brass bell"),
    (("track_light", "recessed_light", "spotlight"), 0.35, 1.0, "lamp housing"),
    (("pedestal_sculpture",), 0.22, 0.0, "polished marble"),
    (("tablet_pos_terminal",), 0.30, 0.0, "screen glass over plastic"),
    (("ballpoint_pen",), 0.35, 0.0, "moulded plastic"),
    (("bench",), 0.55, 0.0, "oiled wood"),
    (("reception_counter", "low_stand", "brochure_stand"), 0.50, 0.0, "lacquered wood"),
    (("partition_panel",), 0.88, 0.0, "matte gallery panel"),
    (("floor",), 0.30, 0.0, "polished stone"),
    (("wall",), 0.90, 0.0, "gallery plaster"),
    (("painting", "print", "photograph"), 0.80, 0.0, "canvas and matte print"),
    (("guestbook", "map_leaflet"), 0.85, 0.0, "paper"),
    # -- the grocery store (scene_102) --
    (("gondola_shelving", "refrigerated_display_case"), 0.35, 1.0, "shop steel shelving"),
    (("shopping_basket", "produce_tray"), 0.45, 0.0, "moulded plastic"),
    (("retail_counter", "display_table"), 0.45, 0.0, "laminate counter"),
    (("display_screen",), 0.10, 0.0, "screen glass"),
    (("specials_board", "bulletin_board"), 0.90, 0.0, "chalkboard and cork"),
    (("logo_sign", "aisle_sign", "price_placard"), 0.55, 0.0, "printed signage"),
    # -- the open-plan office (scene_115) --
    # Appended, so the generic ``floor``/``wall``/``print`` rows above match first.
    (("whiteboard",), 0.12, 0.0, "enamel whiteboard"),
    (("office_desk", "credenza", "water_cooler"), 0.45, 0.0, "melamine and lacquer"),
    (("office_chair", "visitor_chair"), 0.60, 0.0, "mesh back and moulded seat"),
    (("office_divider_panel",), 0.88, 0.0, "fabric screen"),
    (("all_in_one_desktop_computer",), 0.20, 0.0, "screen glass over aluminium"),
    (("smartphone",), 0.15, 0.0, "phone glass"),
    (
        ("wireless_keyboard", "computer_mouse", "speakerphone_puck", "office_stapler",
         "pen_cup_holder", "pencil", "file_organizer", "clock", "smoke_detector"),
        0.38, 0.0, "moulded desk plastic",
    ),
    (
        ("linear_led_ceiling_light", "recessed_downlight", "flush_mount"),
        0.35, 0.0, "painted light housing",
    ),
    (("water_bottle",), 0.15, 0.0, "clear plastic bottle"),
    (("coffee_mug",), 0.20, 0.0, "glazed ceramic"),
    (("potted_plant",), 0.65, 0.0, "leaves and terracotta"),
    (
        ("manila_folder", "spiral_notebook", "spiral_bound_notepad", "sticky_note_pad"),
        0.85, 0.0, "paper and card",
    ),
    # -- the dining room (scene_046) --
    # Every entry carries the ``dining_room_`` prefix, so this block matches
    # no other scene.
    (("dining_room_mirror",), 0.06, 1.0, "silvered glass"),
    (("dining_room_chandelier",), 0.25, 0.0, "frosted glass globes"),
    (("dining_room_rug",), 0.85, 0.0, "wool rug"),
    (
        ("dining_room_sideboard", "dining_room_wine_cabinet",
         "dining_room_round_dining_table", "dining_room_serving_cart"),
        0.45, 0.0, "lacquered wood furniture",
    ),
    (("dining_room_dining_chair",), 0.60, 0.0, "moulded seat on a wood frame"),
    (("dining_room_wine_glass", "dining_room_wine_decanter"), 0.08, 0.0, "stemware"),
    (("dining_room_wine_bottle",), 0.15, 0.0, "bottle glass"),
    (
        ("dining_room_dinner_plate", "dining_room_shallow_ceramic_bowl",
         "dining_room_catchall_bowl", "dining_room_votive_candle_holder"),
        0.20, 0.0, "glazed ceramic",
    ),
    (("dining_room_candle",), 0.55, 0.0, "wax"),
    (("dining_room_leather_coaster",), 0.60, 0.0, "leather"),
    (("dining_room_waiters_corkscrew",), 0.35, 0.0, "moulded handle and steel"),
    (("dining_room_hardcover_book",), 0.75, 0.0, "cloth binding"),
    (
        ("dining_room_potted_snake_plant", "dining_room_potted_succulent",
         "dining_room_potted_pothos_plant"),
        0.65, 0.0, "leaves and terracotta",
    ),
    (
        ("dining_room_eucalyptus_stem", "dining_room_white_flower",
         "dining_room_rose_stem"),
        0.70, 0.0, "cut stems",
    ),
)
"""(name substrings, roughness, metallic, description) rows. First match wins."""


def surface_for(material_name: str, table=SURFACES):
    """``(roughness, metallic, what)`` for a material, or ``None`` to leave it unchanged."""
    for names, roughness, metallic, what in table:
        if any(n in material_name for n in names):
            return roughness, metallic, what
    return None


def _use_metallic_workflow(shader) -> None:
    """Turn off a material's specular workflow.

    UsdPreviewSurface ignores ``metallic`` while the specular workflow is on.
    """
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)


def apply_surfaces(scene_dir: str | Path, overlay: str | Path | None = None,
                   table=SURFACES) -> tuple[Path, int, int]:
    """Author roughness and metallic into a scene's prepared overlay.

    Returns the overlay path, and how many materials were given a surface or left as generated.
    """
    scene_dir = Path(scene_dir)
    dst = Path(overlay).resolve() if overlay else (scene_dir / "scene_static.usda").resolve()
    if not dst.exists():
        raise FileNotFoundError(
            f"no overlay at {dst} -- `prepare` writes it, and must run first"
        )
    layer = Sdf.Layer.FindOrOpen(str(dst))
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(Usd.EditTarget(layer))

    touched = left = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        # Metalness workflow on every material, so ``metallic`` decides reflectance.
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

        # The material is the shader's parent.
        _use_metallic_workflow(shader)
        hit = surface_for(prim.GetParent().GetName(), table)
        if hit is None:
            left += 1
            continue
        roughness, metallic, _ = hit
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        touched += 1

    layer.Save()
    return dst, touched, left


# --- Window daylight texture ---
# Replaces the scene's low-resolution window textures with one generated daylight sheet.

GLASS_TEXTURE = (1024, 512)
"""Size of the daylight sheet generated for a window, in pixels."""


def _daylight(size=GLASS_TEXTURE):
    """A soft top-to-bottom daylight wash, for a window with nothing behind it."""
    from PIL import Image

    w, h = size
    img = Image.new("RGB", (w, h))
    px = img.load()
    top, bottom = (236, 242, 250), (250, 246, 238)
    for y in range(h):
        t = y / max(1, h - 1)
        row = tuple(round(a + (b - a) * t) for a, b in zip(top, bottom))
        for x in range(w):
            px[x, y] = row
    return img


def retexture_glass(scene_dir, overlay=None, match="window"):
    """Replace a scene's window textures with a generated daylight sheet."""
    from PIL import Image  # noqa: F401

    scene_dir = Path(scene_dir)
    dst = Path(overlay).resolve() if overlay else (scene_dir / "scene_static.usda").resolve()
    layer = Sdf.Layer.FindOrOpen(str(dst))
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(Usd.EditTarget(layer))
    out_dir = scene_dir / "artwork"
    out_dir.mkdir(exist_ok=True)
    sheet = out_dir / "daylight.png"
    if not sheet.exists():
        _daylight().save(sheet)
    done = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        if match not in prim.GetParent().GetName():
            continue
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(sheet.resolve())))
        done += 1
    layer.Save()
    return dst, done, sheet
