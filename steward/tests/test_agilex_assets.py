"""Bundled AgileX assets load after relocation without a USD decoder or local import cache."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from steward.assets.agilex import agilex_usd
from test_agilex import imported_profile  # noqa: F401 - shared pytest fixture

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / 'assets/agilex'


def test_bundled_files_match_the_source_and_conversion_record():
    report = json.loads((ASSETS / 'runtime/report.json').read_text())
    assert hashlib.sha256((ASSETS / 'robot.usd').read_bytes()).hexdigest() == report['sha256']
    files = report['files']
    assert set(files) == {p.name for p in (ASSETS / 'runtime').iterdir() if p.name != 'report.json'}
    for name, digest in files.items():
        assert hashlib.sha256((ASSETS / 'runtime' / name).read_bytes()).hexdigest() == digest
    template = json.loads((ASSETS / 'runtime/template.json').read_text())
    refs = {g['ref'] for link in template['body']['links'] for gs in link.get('geometry', {}).values()
            for g in gs if g['kind'] == 'mesh'}
    assert refs == {name for name in files if name.endswith('.msh')}
    assert all(Path(ref).name == ref for ref in refs)


@pytest.mark.parametrize('absolute', [False, True])
def test_mesh_paths_follow_the_template_and_missing_meshes_fail_early(imported_profile, absolute, monkeypatch):
    template = imported_profile.parent / 'template.json'
    entity = json.loads(template.read_text())
    mesh = template.parent / 'meshes/part.msh'
    mesh.parent.mkdir()
    mesh.write_bytes(b'path resolution only')
    base = next(link for link in entity['body']['links'] if link['id'].endswith('/base_link'))
    base['geometry']['visual'] = [
        {'kind': 'mesh', 'ref': str(mesh) if absolute else 'meshes/part.msh'}]
    template.write_text(json.dumps(entity))
    monkeypatch.chdir(mesh.parent)
    model = agilex_usd(imported_profile)
    refs = [g['ref'] for link in model['body']['links'] for gs in link.get('geometry', {}).values()
            for g in gs if g['kind'] == 'mesh']
    assert refs and all(ref == str(mesh) for ref in refs)
    mesh.unlink()
    with pytest.raises(ValueError, match='USD embodiment mesh missing'):
        agilex_usd(imported_profile)


def test_relocated_checkout_builds_the_default_body_without_source_usd(tmp_path):
    checkout = tmp_path / 'checkout/steward'
    shutil.copytree(ROOT / 'src', checkout / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'config', checkout / 'config')
    shutil.copytree(ASSETS / 'runtime', checkout / 'assets/agilex/runtime')
    env = {k: v for k, v in os.environ.items() if k != 'STEWARD_AGILEX_USD_CONFIG'}
    env['PYTHONPATH'] = str(checkout / 'src')
    result = subprocess.run([sys.executable, '-c', '''
import importlib.abc
import json
from pathlib import Path
import sys
class NoUsd(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pxr', 'mujoco_usd'):
            raise AssertionError('runtime must not import the USD decoder')
sys.meta_path.insert(0, NoUsd())
import mujoco
import steward
from steward.ego.bodies import prepare_body
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, SceneState
root = Path(sys.argv[1])
assert Path(steward.__file__).is_relative_to(root / 'src')
assert not (root / 'assets/agilex/robot.usd').exists()
assert not (root / 'var').exists()
setup = prepare_body('agilex-usd')
for link in setup.model['body']['links']:
    for geometries in link.get('geometry', {}).values():
        for geom in geometries:
            if geom['kind'] == 'mesh':
                mesh = Path(geom['ref'])
                assert mesh.is_relative_to(root / 'assets/agilex/runtime') and mesh.is_file()
build = build_mjcf(SceneState(mode='world', entities=[Entity.from_dict(setup.model)]))
model = build.compile()
data = mujoco.MjData(model)
build.apply_initial(model, data)
for _ in range(100):
    mujoco.mj_step(model, data)
assert not any(w.number for w in data.warning)
print(json.dumps({'meshes': model.nmesh, 'geometries': model.ngeom, 'steps': 100}))
''', str(checkout)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['meshes'] > 0
