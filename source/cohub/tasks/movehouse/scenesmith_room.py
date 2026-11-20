"""The house this table is carried across: a downloaded SceneSmith scene.

The task's origin is the centre of the doorway between the bedroom and the living room.

    python scripts/setup_assets.py rooms    # fetches and prepares it
"""

from cohub.tasks._base.props import ceiling_cfg
from cohub.scenesmith.cfg import Move, Widen, scene_data_dir, scenery_cfg_at

SCENE_ID = "House/scene_098"
"""A four-room home: 161 objects, 7.3 x 8.2 x 2.7 m, floor top at z=0, no ceiling."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``scripts/setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 270.0
"""Room yaw [degrees]: task_x = scene_y - 3.600, task_y = 3.378 - scene_x."""

ROOM_ORIGIN_AT = (3.378, 3.600)
"""The scene coordinate that becomes the task's origin: the doorway's centre [m]."""

SHELF = (
    "living_room_bookshelf_0$",
    "living_room_board_game_box_0$",
    "living_room_board_game_box_1$",
    "living_room_board_game_box_2$",
    "living_room_board_game_box_2_s1_0$",
    "living_room_board_game_box_3$",
    "living_room_board_game_box_4$",
    "living_room_board_game_box_5$",
    "living_room_board_game_box_6$",
    "living_room_card_game_box_0$",
    "living_room_card_game_box_1$",
    "living_room_card_game_box_2$",
    "living_room_hardcover_book_0$",
    "living_room_hardcover_book_1$",
    "living_room_hardcover_book_2$",
    "living_room_hardcover_book_3$",
    "living_room_hardcover_book_5$",
    "living_room_hardcover_book_3_s1_1$",
    "living_room_hardcover_book_3_s2_0$",
    "living_room_hardcover_book_3_s3_0$",
    "living_room_paperback_book_0$",
    "living_room_paperback_book_1$",
    "living_room_paperback_book_2$",
    "living_room_paperback_book_3$",
    "living_room_journal_0_s1_2$",
    "living_room_journal_0_s2_1$",
    "living_room_journal_0_s3_1$",
    "living_room_comic_book_0$",
    "living_room_comic_book_1$",
    "living_room_comic_book_2$",
    "living_room_comic_book_3$",
    "living_room_reference_book_0$",
    "living_room_reference_book_1$",
    "living_room_reference_book_2$",
    "living_room_art_book_0$",
    "living_room_art_book_1$",
    "living_room_dictionary_0$",
    "living_room_dictionary_1$",
    "living_room_textbook_0$",
    "living_room_textbook_1$",
    "living_room_lifestyle_magazine_0$",
    "living_room_home_decor_magazine_0$",
    "living_room_picture_frame_0$",
    "living_room_potted_succulent_plant_0$",
)
"""The living room bookshelf and what stands on it. Every entry is anchored with ``$``."""

HIDDEN = (
    # the bedroom dresser and what stood on it
    "bedroom_dresser_0",
    "bedroom_jewelry_tray_0",
    "bedroom_scented_candle_0",
    "bedroom_perfume_bottle_0",
    # in the carry lane or in Robot A's place
    "bedroom_wardrobe_0",
    "bedroom_writing_desk_0",
    "bedroom_desk_chair_0",
    "bedroom_laptop_0",
    "bedroom_spiral_notebook_0",
    "bedroom_coffee_mug_0",
    "bedroom_ballpoint_pen_0",
    "bedroom_floor_lamp_1",
    "bedroom_nightstand_1",
    "bedroom_smartphone_1",
    "bedroom_glass_tumbler_0",
    "bedroom_paperback_novel_1",
    "living_room_ottoman_0",
) + SHELF
"""Prims removed from the house."""

REARRANGED = (
    # the seating group, moved together toward the far wall
    Move(
        names=(
        "living_room_couch_0",
        "living_room_folded_quilted_blanket_0",
        "living_room_home_decor_magazine_1",
        "living_room_coffee_table_0",
        "living_room_tv_remote_control_0",
        "living_room_lifestyle_magazine_0_s0_0",
        "living_room_bud_vase_0",
        "living_room_cork_drink_coaster_0",
        "living_room_home_decor_magazine_0_s0_1",
        "living_room_side_table_0",
        "living_room_tv_remote_control_1",
        "living_room_cork_drink_coaster_0_s4_0",
        "living_room_coffee_mug_0_s4_1",
        ),
        dy=-1.0471,
    ),
    # The doorway, cut 0.10 m wider on each jamb. Both wall shells of the partition are named.
    Widen(
        names=(
            "room_geometry_bedroom_room_geometry_body_link_north_wall_visual_visual",
            "room_geometry_bedroom_room_geometry_body_link_north_wall_collision_0_collision",
            "room_geometry_bedroom_room_geometry_body_link_north_wall_collision_1_collision",
            "room_geometry_living_room_room_geometry_body_link_south_wall_visual_visual",
            "room_geometry_living_room_room_geometry_body_link_south_wall_collision_1_collision",
            "room_geometry_living_room_room_geometry_body_link_south_wall_collision_0_collision",
        ),
        across=(-0.4496, 0.4504),
        to=(-0.5496, 0.5504),
    ),
)
"""Edits to the room: the seating group moved, and the doorway widened from 0.90 to 1.10 m."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID,
    rearrange=REARRANGED,
)


SCENE_CAM_POS = (-1.40, 3.20, 2.40)
SCENE_CAM_ROT = (0.16104736, 0.16104736, -0.68852287, 0.68852287)
SCENE_CAM_FOCAL = 10.5
"""Third-person camera in the bedroom: position [m], rotation (x, y, z, w) and focal length."""

SCENE_CAM_2_POS = (4.30, -1.40, 2.40)
SCENE_CAM_2_ROT = (-0.22606265, 0.07125997, 0.92655903, 0.29207201)
"""Second third-person camera, in the living room: position [m] and rotation (x, y, z, w), looking at (1.30, 0.70, 0.50)."""

CEILING_SIZE = (8.2, 7.31)
CEILING_CENTRE = (0.45, -0.227)
CEILING_TOP_Z = 2.6
"""Footprint, centre and top height of the ceiling slab, in the task frame [m]."""

CEILING_CFG = ceiling_cfg(CEILING_SIZE, CEILING_CENTRE, CEILING_TOP_Z)
