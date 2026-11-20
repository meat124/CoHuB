"""Dimensions of the painting, the hook, the hang wall and the robots' stance.

Two G1s lift a framed painting off the floor and hang it on a single wall peg.
Each robot holds one side rail with one hand. ``data/*.usda`` are authored by
hand against these values, so change both together.
"""

import math

# Frame
ASSET_SIZE = (0.603030, 0.019993, 0.464126)
"""Bounds of Poly Haven's ``fancy_picture_frame_01`` as downloaded, landscape [m]."""

FRAME_SCALE = 1.90
"""Scale applied to the frame asset."""

FRAME_WIDTH = ASSET_SIZE[0] * FRAME_SCALE
FRAME_HEIGHT = ASSET_SIZE[2] * FRAME_SCALE
FRAME_DEPTH = ASSET_SIZE[1] * FRAME_SCALE
FRAME_HALF_WIDTH = FRAME_WIDTH / 2
FRAME_HALF_HEIGHT = FRAME_HEIGHT / 2
FRAME_HALF_DEPTH = FRAME_DEPTH / 2

FRAME_MASS = 4.0
"""Mass of the picture [kg]."""

FRAME_CONTACT_OFFSET = 0.006

# Hanging loop: stands above the top rail, as deep as the frame, so the hook
# stays visible.
LOOP_RISE = 0.042
"""Height of the loop's opening above the frame's top rail [m]."""
LOOP_Z = FRAME_HALF_HEIGHT + LOOP_RISE
"""Height of the cap's underside, which bears on the hook [m, picture frame]."""
LOOP_INNER_WIDTH = 0.052
"""Clear opening across the loop [m]."""

# Grip: each robot takes a side rail, GRIP_RISE above the picture's centre.
GRIP_RISE = 0.36 * FRAME_HEIGHT
"""Height of the grip above the picture's centre [m]."""
GRIP_Y = 0.0
"""The y of the grip in the picture's frame [m]: on its centre plane."""

GRIP_SITES = (
    (-FRAME_HALF_WIDTH, GRIP_Y, GRIP_RISE),
    (FRAME_HALF_WIDTH, GRIP_Y, GRIP_RISE),
)
"""Grip sites, one per robot, on the frame's side rails [m, picture frame]."""

# Hook: a straight peg with a groove where the loop seats and a knob on the end.
HOOK_SHANK_RADIUS = 0.014
HOOK_SHANK_LENGTH = 0.150
"""Radius and length of the peg's shank [m]."""
HOOK_GROOVE_RADIUS = 0.008
"""Radius of the groove the loop seats in [m]."""
HOOK_SEAT_DEPTH = 0.090
"""Distance from the wall's face to the seat [m]. The picture hangs this far off the wall."""
HOOK_KNOB_RADIUS = 0.017
"""Radius of the knob on the end of the peg [m]."""
HOOK_PLATE_PROUD = 0.010
"""How far the mounting plate's face stands out from the wall [m]."""

FRAME_LEAN_DEG = 0.0
"""Lean of the resting picture [deg]. Zero: it stands upright."""

FRAME_STANDOFF = 0.15
"""Gap between the wall's face and the back of the resting picture [m]."""

HOOK_BASE_HEIGHT = 1.10
"""Height of the peg [m]. The ego camera's view limits it to about 1.12 m."""

HOOK_RAISE = 0.0
"""Extra height of the peg above ``HOOK_BASE_HEIGHT`` [m]."""

HOOK_HEIGHT = round(HOOK_BASE_HEIGHT + HOOK_RAISE, 3)
"""Height of the hook above the floor [m], before the reset draw."""

HOOK_X_JITTER = 0.050
"""Reset range of the hook's position along the wall, +- [m]."""
HOOK_Z_RANGE = (-0.040, 0.0)
"""Reset range of the hook's height [m]. Downward only."""

# Layout
WALL_Y = 1.80
"""The y of the hang wall's face [m]. The room is on its -Y side."""

WALL_Y_JITTER = 0.025
"""Reset range of the hang wall's y, +- [m]."""
HUNG_CENTRE_Y = WALL_Y - HOOK_SEAT_DEPTH
"""The y of a hung painting's centre plane [m]."""
FRAME_HUNG_Z = HOOK_HEIGHT + HOOK_GROOVE_RADIUS - LOOP_Z
"""Height of a hung painting's centre [m]."""

STANCE_DX = 0.78
"""Half the distance between the two robots [m]."""
STANCE_YAW_DEG = 36.0
"""Inward turn of each robot toward the other [deg]."""
GRIP_HAND_FWD = 0.40
"""How far ahead of the pelvis the gripping hand works [m]."""

EGO_FORWARD_OFFSET = 0.19
EGO_HEIGHT_ABOVE_ROOT = 0.433
"""Position of the ego lens relative to the pelvis root: forward and up [m]."""

FRAME_PLANE_Y = HUNG_CENTRE_Y
GRIP_PLANE_Y = FRAME_PLANE_Y + GRIP_Y
"""The y of the plane the hands grip in [m]."""
ROBOT_Y = GRIP_PLANE_Y - GRIP_HAND_FWD

# Reset: the painting starts standing on the floor under the peg.
_LEAN = math.radians(FRAME_LEAN_DEG)

FRAME_START_Z = (FRAME_HALF_HEIGHT * math.cos(_LEAN)) + (FRAME_HALF_DEPTH * math.sin(_LEAN))
"""Height of the resting picture's centre [m]."""

FRAME_START_Y = (
    WALL_Y - FRAME_STANDOFF - FRAME_HALF_DEPTH / math.cos(_LEAN) - FRAME_HALF_HEIGHT * math.sin(_LEAN)
)
"""The y of the resting picture's centre [m]."""

FRAME_START_ROT = (-math.sin(_LEAN / 2), 0.0, 0.0, math.cos(_LEAN / 2))
"""Rotation of the resting picture, (x, y, z, w). Negative x tips its top toward the wall (+Y)."""

FRAME_LIFT_HEIGHT = FRAME_HUNG_Z - FRAME_START_Z
"""Rise of the painting's centre from its rest on the floor to its hung height [m]."""
FRAME_CLEAR_RISE = FRAME_LIFT_HEIGHT + HOOK_SHANK_RADIUS - HOOK_GROOVE_RADIUS
"""Rise the painting needs to clear the shank before it is lowered into the groove [m]."""

FRAME_RESET_X_JITTER = 0.030
FRAME_RESET_Y_JITTER = 0.010
"""Reset range of the painting's position on the floor, +- [m], on top of the station's draw."""

# Gallery
ROOM_HALF_X = 6.50
ROOM_HEIGHT = 4.05
"""Half-width and height of the hang wall [m]. It covers the gallery's south wall."""
WALL_THICKNESS = 0.12
BASEBOARD_HEIGHT = 0.11

# Guards
FRAME_DROP_HEIGHT = FRAME_START_Z - 0.15
"""Height of the painting's centre below which it counts as dropped [m]."""


def eye_position(sign: int = -1) -> tuple[float, float, float]:
    """Where one operator's ego lens sits at the hang pose [m].

    ``sign`` selects the robot: -1 is the one on -X. The lens rides on
    ``torso_link``, ``EGO_FORWARD_OFFSET`` ahead of the root along the heading.
    """
    yaw = math.radians(90.0 + sign * STANCE_YAW_DEG)
    return (
        sign * STANCE_DX + EGO_FORWARD_OFFSET * math.cos(yaw),
        ROBOT_Y + EGO_FORWARD_OFFSET * math.sin(yaw),
        0.725 + EGO_HEIGHT_ABOVE_ROOT,
    )


def eye_height() -> float:
    """Where the ego lens sits above the floor, standing [m]."""
    return eye_position()[2]
