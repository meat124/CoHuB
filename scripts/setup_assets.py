"""Fetch what the benchmark loads and git does not carry.

    python scripts/setup_assets.py rooms
    python scripts/setup_assets.py isaac --dest isaac_assets/Isaac

``rooms`` downloads and prepares every SceneSmith scene the tasks stand in
(needs ``pxr``, no Kit). ``isaac`` mirrors the Isaac cloud assets an evaluation
loads from disk, for ``ISAAC_ASSETS`` (standard library only).
"""

import argparse
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from pathlib import Path

# Rooms land inside the checkout, in a directory git ignores.
SCENES_DIR = Path(__file__).resolve().parents[1] / "external" / "scenesmith_data" / "scenes"
SCENES_DATASET = "nepfaff/scenesmith-example-scenes"
SCENES_URL = f"https://huggingface.co/datasets/{SCENES_DATASET}/resolve/main"

ISAAC_BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ISAAC_ROOT = "Assets/Isaac/6.0/Isaac"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Prefixes to mirror, relative to ISAAC_ROOT.
ISAAC_PREFIXES = [
    "Robots/Unitree/G1/",                                            # G1_29DOF_CFG
    "Environments/Grid/",                                            # GroundPlaneCfg default
    "IsaacLab/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/",  # Pink IK URDF + meshes
    "IsaacLab/Policies/Agile/",                                      # agile_locomotion.pt
]

# Thumbnails are not needed.
SKIP_PARTS = (".thumbs/",)

ISAAC_REQUIRED = [
    "Robots/Unitree/G1/g1.usd",
    "Environments/Grid/default_environment.usd",
    "IsaacLab/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/"
    "g1_29dof_with_hand_only_kinematics.urdf",
    "IsaacLab/Policies/Agile/agile_locomotion.pt",
]


def _download(url: str, dst: Path, attempts: int = 8) -> Path:
    """Fetch url into dst, resuming with a Range request after a short read.

    Bytes go to a .part file until their count matches Content-Length."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    print(f"fetching {url}")
    done = part.stat().st_size if part.exists() else 0
    total = 0
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(request) as response, open(part, "ab") as out:
                if done and response.status != 206:      # no Range support: start over
                    out.seek(0); out.truncate(); done = 0
                length = int(response.headers.get("Content-Length") or 0)
                total = done + length if length else total
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
            print(f"\n  attempt {attempt + 1}: {err}")
        print()
        if not total or done >= total:
            break
        print(f"  short by {(total - done) / 1e6:.1f} MB; resuming ({attempt + 1}/{attempts})")
    if total and done != total:
        raise SystemExit(f"{url}: got {done} of {total} bytes after {attempts} attempts; rerun to resume")
    part.replace(dst)
    return dst


def fetch_scene(scene: str, root: Path) -> Path:
    """Download and unpack one published scene, e.g. ``Room/scene_084``. Returns its directory."""
    subset, scene_id = scene.split("/", 1)
    out_dir = root / scene_id
    tar_path = root / f"{scene_id}.tar"

    # The unpacked directory decides, not the tar.
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{out_dir} already unpacked")
    else:
        if not tar_path.exists():
            _download(f"{SCENES_URL}/{subset}/{scene_id}.tar", tar_path)
        else:
            print(f"{tar_path} already here")
        # Unpack into a staging directory and rename at the end, so a
        # half-unpacked directory never passes the test above.
        staging = out_dir.with_name(out_dir.name + ".unpacking")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        print(f"unpacking into {out_dir}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(staging)
        if out_dir.exists():
            out_dir.rmdir()
        staging.rename(out_dir)
        tar_path.unlink()
        print(f"removed {tar_path.name} ({out_dir.name} is the copy that is used)")
    return out_dir


def cmd_rooms(args: argparse.Namespace) -> int:
    """Fetch and prepare every scene the shipped tasks need."""
    # Imported here so that `isaac` runs without pxr or this checkout's package.
    from cohub.scenesmith import REQUIRED_SCENES, SURFACED_SCENES
    from cohub.scenesmith.prepare import retexture_glass, prepare, apply_surfaces

    root = Path(args.data_dir) if args.data_dir else SCENES_DIR
    for scene, dynamic in REQUIRED_SCENES.items():
        print(f"=== {scene} ===")
        scene_dir = fetch_scene(scene, root)
        out, pinned, free, joints = prepare(scene_dir, dynamic=list(dynamic))
        print(f"prepared {out.name}: {pinned} pinned, {joints} joints frozen, "
              f"{free} simulated")
        if scene in SURFACED_SCENES:
            _, touched, left = apply_surfaces(scene_dir)
            print(f"surfaced {out.name}: {touched} materials given a surface, "
                  f"{left} left as generated")
            _, panes, _ = retexture_glass(scene_dir)
            print(f"glazed   {out.name}: {panes} window panes")
    print("\nready. The tasks will find their rooms.")
    return 0


def list_prefix(prefix: str) -> list[tuple[str, int]]:
    """Every key under `prefix`, following continuation tokens."""
    keys: list[tuple[str, int]] = []
    token = None
    while True:
        query = {"list-type": "2", "prefix": f"{ISAAC_ROOT}/{prefix}", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{ISAAC_BUCKET}/?{urllib.parse.urlencode(query)}", timeout=60) as response:
            tree = ET.fromstring(response.read())
        for item in tree.findall("s3:Contents", S3_NS):
            key = item.find("s3:Key", S3_NS).text
            size = int(item.find("s3:Size", S3_NS).text)
            if not any(part in key for part in SKIP_PARTS):
                keys.append((key, size))
        if tree.findtext("s3:IsTruncated", namespaces=S3_NS) != "true":
            return keys
        token = tree.findtext("s3:NextContinuationToken", namespaces=S3_NS)


def mirror_file(key: str, size: int, dest_root: Path) -> bool:
    """Download one key unless a local copy of the same size exists. Returns True if fetched."""
    target = dest_root / key[len(ISAAC_ROOT) + 1:]
    if target.is_file() and target.stat().st_size == size:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ISAAC_BUCKET}/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url, timeout=300) as response, open(target, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    return True


def cmd_isaac(args: argparse.Namespace) -> int:
    """Mirror ISAAC_PREFIXES into ``--dest``. Returns 1 when a required file is missing."""
    dest = Path(args.dest).expanduser().resolve()

    total_bytes = 0
    for prefix in ISAAC_PREFIXES:
        keys = list_prefix(prefix)
        if not keys:
            print(f"[mirror] WARNING: nothing under {prefix} -- the layout may have moved")
            continue
        prefix_bytes = sum(size for _, size in keys)
        print(f"[mirror] {prefix}: {len(keys)} files, {prefix_bytes / 1e6:.1f} MB")
        fetched = 0
        for key, size in keys:
            if mirror_file(key, size, dest):
                fetched += 1
        total_bytes += prefix_bytes
        print(f"[mirror]   {fetched} downloaded, {len(keys) - fetched} already present")

    print(f"[mirror] done -> {dest} ({total_bytes / 1e6:.1f} MB)")
    missing = [name for name in ISAAC_REQUIRED if not (dest / name).exists()]
    for name in ISAAC_REQUIRED:
        print(f"[mirror]   {'OK  ' if (dest / name).exists() else 'MISS'} {name}")
    return 1 if missing else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rooms", help="fetch and prepare the rooms the tasks stand in")
    r.add_argument("--data-dir", default=None, help="override where scenes go")
    r.set_defaults(func=cmd_rooms)

    i = sub.add_parser("isaac", help="mirror the Isaac assets an evaluation loads from disk")
    i.add_argument("--dest", required=True, help="ISAAC_ASSETS root to write")
    i.set_defaults(func=cmd_isaac)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
