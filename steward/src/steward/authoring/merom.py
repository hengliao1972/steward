"""Install the bundled Merom house and a living-room robot into a Steward data root.

Mesh names are content hashes. Published scenes refer to the extracted files in the data root,
so loading a built scene does not depend on the source USD or the checkout's asset directory.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

from steward.assets.library import instantiate
from steward.embodiments import robot_entity
from steward.math3d import quat_from_yaw

if TYPE_CHECKING:
    from steward.authoring.service import AuthoringService


BUNDLE = Path(__file__).resolve().parents[1] / "assets" / "merom_house"


def merom_house_spec() -> dict:
    """Return the default robot placement in the living room, facing the sofa.

    :returns: Entity placements with a registered body and metre/radian coordinates.
    """
    return {"entities": [{"id": "w:robot-1", "asset": "cobot-magic",
                          "at": {"x": 1.7, "y": 7.2, "yaw": math.pi / 2}}]}


def build_merom_house(service: AuthoringService) -> dict:
    """Install the house with separate collision pieces and a Cobot Magic in the living room.

    :param service: Authoring service whose data root receives the scene and meshes.
    :returns: Published scene name and scenario record.
    :raises ValueError: A bundled mesh does not match its content-hash filename.
    """
    scene = json.loads((BUNDLE / "scene.json").read_text(encoding="utf-8"))
    scenario = json.loads((BUNDLE / "scenario.json").read_text(encoding="utf-8"))
    geometries = [g for ent in scene["entities"] for link in ent["body"]["links"]
                  for group in link["geometry"].values() for g in group if g["kind"] == "mesh"]
    with ZipFile(BUNDLE / "meshes.zip") as archive:
        meshes = {ref: archive.read(ref) for ref in {g["ref"] for g in geometries}}
    for ref, content in meshes.items():
        if ref != hashlib.sha256(content).hexdigest() + ".msh":
            raise ValueError(f"Merom mesh {ref!r} does not match its content hash")

    mesh_dir = service.root.resolve() / "imports" / "merom-house"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for ref, content in meshes.items():
        (mesh_dir / ref).write_bytes(content)
    for geom in geometries:
        geom["ref"] = str(mesh_dir / geom["ref"])
    for placement in merom_house_spec()["entities"]:
        at = placement["at"]
        robot = instantiate(robot_entity(placement["asset"]), placement["id"],
                            {"p": [at["x"], at["y"], 0], "q": quat_from_yaw(at["yaw"])},
                            asset={"id": placement["asset"]})
        scene["entities"].append(robot.to_dict())
    scene_path = service.root / "scenes" / f"{scenario['scene']}.json"
    scene_path.write_text(json.dumps(scene, ensure_ascii=False) + "\n", encoding="utf-8")
    published = service.save_scenario(scenario)
    return {"scene": scenario["scene"], "scenario": published}
