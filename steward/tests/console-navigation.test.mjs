/** Console regressions exercise the shipped camera handlers and label overlays. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../src/steward/world/console.html', import.meta.url), 'utf8');
const cameraScript = html.slice(html.indexOf('function cameraFrame('), html.indexOf('function project('));
const script = html.slice(html.indexOf('let viewPosting ='), html.indexOf('// ------------------------------------------------------------------ refresh'));
assert.ok(script.includes('function postView()') && script.includes("vp.addEventListener('keydown'"));

test('World and Ego labels name house objects even when their task role is unknown', () => {
  const world = { labels: [], clearRect() {} }, ego = { labels: [], clearRect() {} };
  const objects = ['table', 'sofa', 'bread'].map((category, i) => ({
    id: `e:${i + 1}`, category, role: i === 0 ? 'unknown' : category,
    center: [i * 2, 0, 0.5], size: [1, 1, 1], existence: 1,
  }));
  const groundTruth = objects.map(o => ({ ...o, id: `w:${o.category}`, role: 'object' }));
  groundTruth.push({ id: 'w:floor/floor', category: 'floor', role: 'environment' });
  const controls = {
    ovWorld: { getContext: () => world }, ovEgo: { getContext: () => ego },
    showGtLabels: { checked: true }, showEgoLabels: { checked: true },
    showSigma: { checked: false }, showError: { checked: false },
  };
  const context = vm.createContext({
    $: id => controls[id], state: { loaded: true, groundTruth, robots: { robot: { ego: { objects } } } },
    robotId: 'robot', panes: { world: { rendered: {} }, ego: { rendered: {} } },
    cameraFrame: camera => camera,
    painter: ctx => ({ text: (_point, text) => ctx.labels.push(text), flush() {} }),
  });
  vm.runInContext(html.slice(html.indexOf('const NAMES ='), html.indexOf('const RENDER_W ='))
    + html.slice(html.indexOf('const label ='), html.indexOf('// ------------------------------------------------------------------ renders')),
  context);
  vm.runInContext('drawOverlays()', context);
  const expected = JSON.parse(readFileSync(new URL('expected/console-labels.json', import.meta.url), 'utf8'));
  assert.deepEqual({ world: world.labels, ego: ego.labels }, expected);
  const scene = JSON.parse(readFileSync(new URL('../src/steward/assets/merom_house/scene.json', import.meta.url), 'utf8'));
  for (const entity of scene.entities) {
    context.category = entity.category.value;
    assert.ok(vm.runInContext('NAMES[category]', context), `missing label for ${entity.category.value}`);
  }
});

function harness(t, { azimuth = 90, fps = 10, api } = {}) {
  let now = 0, nextFrame = 0;
  const frames = new Map(), posts = [], errors = [];
  const document = new EventTarget(), window = new EventTarget();
  document.hidden = false;
  const presets = ['south', 'southeast', 'southwest', 'top'].map(name => ({
    dataset: { view: name }, classList: { remove() {}, toggle() {} },
  }));
  document.querySelectorAll = () => presets;
  const panels = Object.fromEntries(['vpWorld', 'vpEgo'].map(id => {
    const panel = new EventTarget();
    panel.focus = () => {
      if (document.activeElement !== panel) document.activeElement?.blur();
      document.activeElement = panel;
    };
    panel.blur = () => { document.activeElement = null; panel.dispatchEvent(new Event('blur')); };
    panel.closest = () => null;
    panel.setPointerCapture = () => {};
    panel.classList = { add() {}, remove() {} };
    return [id, panel];
  }));
  const context = vm.createContext({
    document, window, performance: { now: () => now },
    requestAnimationFrame: fn => { const id = ++nextFrame; frames.set(id, fn); return id; },
    cancelAnimationFrame: id => frames.delete(id),
    $: id => panels[id], rad: deg => deg * Math.PI / 180,
    view: { lookat: [0, 0, 0.5], distance: 8, azimuth, elevation: -50 },
    state: { loaded: true }, fps, RENDER_W: 1024, RENDER_H: 768,
    panes: { world: {}, ego: {} }, drawOverlays() {},
    showError: message => errors.push(message),
    api: (...args) => { posts.push(structuredClone(args)); return api?.(...args); },
  });
  vm.runInContext(cameraScript + script, context);
  t.after(() => vm.runInContext('stopNavigation(false)', context));
  const event = (target, type, fields = {}) => {
    const e = new Event(type, { cancelable: true });
    Object.assign(e, { shiftKey: false, button: 0, pointerId: 1, ...fields });
    target.dispatchEvent(e);
    return e;
  };
  const key = (code, fields = {}, panel = panels.vpWorld) => event(panel, 'keydown', { code, ...fields });
  const step = async ms => {
    now += ms;
    const pending = [...frames.values()];
    frames.clear();
    for (const frame of pending) frame(now);
    // Flush the real postView promise and its single coalesced successor.
    await Promise.resolve(); await Promise.resolve();
  };
  const advance = async (ms, frameMs = 20) => {
    for (let elapsed = 0; elapsed < ms; elapsed += frameMs) await step(Math.min(frameMs, ms - elapsed));
  };
  return { context, panels, document, window, posts, errors, frames, key, event, step, advance,
    release: code => event(window, 'keyup', { code }),
    camera: () => JSON.parse(vm.runInContext('JSON.stringify(view)', context)),
  };
}

function close(actual, expected) { assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`); }

function eye(h) { return JSON.parse(vm.runInContext('JSON.stringify(cameraFrame(view).pos)', h.context)); }

test('dragging after WASD turns in place instead of orbiting the translated target', async t => {
  for (const panelId of ['vpWorld', 'vpEgo']) {
    const h = harness(t), panel = h.panels[panelId];
    h.key('KeyD', {}, panel); await h.advance(500); h.release('KeyD');
    const position = eye(h);
    h.event(panel, 'pointerdown', { clientX: 100, clientY: 100 });
    h.event(panel, 'pointermove', { clientX: 200, clientY: 60 });
    eye(h).forEach((v, i) => close(v, position[i]));
    close(h.camera().azimuth, 55); close(h.camera().elevation, -38);
    assert.equal(h.camera().distance, 8);
    h.event(panel, 'pointerup');
  }
});

test('turning while walking retains the new position and subsequent movement follows the heading', async t => {
  const h = harness(t), panel = h.panels.vpWorld;
  h.event(panel, 'pointerdown', { clientX: 0, clientY: 0 });
  h.key('KeyW'); await h.advance(500);
  const position = eye(h);
  h.event(panel, 'pointermove', { clientX: 100, clientY: 0 });
  eye(h).forEach((v, i) => close(v, position[i]));
  await h.advance(500);
  const walked = eye(h), heading = 55 * Math.PI / 180;
  close(walked[0] - position[0], Math.cos(heading));
  close(walked[1] - position[1], Math.sin(heading));
  close(walked[2], position[2]);
  h.event(panel, 'pointermove', { clientX: 120, clientY: -20 });
  eye(h).forEach((v, i) => close(v, walked[i]));
});

test('looking above the horizon keeps the camera position and the displayed frame overlays', t => {
  const h = harness(t), panel = h.panels.vpWorld;
  const rendered = h.camera();
  h.context.panes.world.rendered = h.context.panes.ego.rendered = rendered;
  const position = eye(h);
  h.event(panel, 'pointerdown', { clientX: 0, clientY: 0 });
  h.event(panel, 'pointermove', { clientX: 0, clientY: -300 });
  close(h.camera().elevation, 40);
  eye(h).forEach((v, i) => close(v, position[i]));
  assert.equal(h.context.panes.world.rendered, rendered);
  assert.equal(h.context.panes.ego.rendered, rendered);
});

test('an interrupted drag cannot turn the camera on a later pointer move', t => {
  for (const stop of [
    h => h.event(h.panels.vpWorld, 'lostpointercapture'),
    h => h.panels.vpWorld.blur(),
    h => h.event(h.window, 'blur'),
    h => { h.document.hidden = true; h.event(h.document, 'visibilitychange'); },
  ]) {
    const h = harness(t), panel = h.panels.vpWorld;
    h.event(panel, 'pointerdown', { clientX: 0, clientY: 0 });
    h.event(panel, 'pointermove', { clientX: 100, clientY: -40 });
    stop(h);
    const before = h.camera();
    h.event(panel, 'pointermove', { clientX: 200, clientY: -80 });
    assert.deepEqual(h.camera(), before);
  }
});

test('WASD follows the view heading, keeps height and zoom, and has equal diagonal speed', async t => {
  for (const [code, x, y] of [['KeyW', 0, 2], ['KeyS', 0, -2], ['KeyA', -2, 0], ['KeyD', 2, 0]]) {
    const h = harness(t);
    assert.ok(h.key(code).defaultPrevented);
    await h.advance(1000);
    close(h.camera().lookat[0], x); close(h.camera().lookat[1], y);
    assert.equal(h.camera().lookat[2], 0.5);
    assert.equal(h.camera().distance, 8);
    assert.equal(h.camera().elevation, -50);
  }
  const rotated = harness(t, { azimuth: 0 });
  rotated.key('KeyW'); rotated.key('KeyD');
  await rotated.advance(1000);
  close(rotated.camera().lookat[0], Math.SQRT2);
  close(rotated.camera().lookat[1], -Math.SQRT2);
});

test('movement uses elapsed time, Shift accelerates, and opposing keys cancel', async t => {
  for (const frameMs of [10, 50]) {
    const h = harness(t);
    h.key('KeyW', { shiftKey: true });
    await h.advance(1000, frameMs);
    close(h.camera().lookat[1], 6);
    h.release('ShiftLeft');
    await h.advance(1000, frameMs);
    close(h.camera().lookat[1], 8);
    h.key('KeyS');
    await h.advance(1000, frameMs);
    close(h.camera().lookat[1], 8);
  }
  const delayed = harness(t);
  delayed.key('KeyW'); await delayed.step(5000);
  close(delayed.camera().lookat[1], 0.2);
});

test('release, focus loss, hidden page, Escape and page exit cancel held movement', async t => {
  for (const stop of [
    h => h.release('KeyW'), h => h.panels.vpWorld.blur(),
    h => h.event(h.window, 'blur'), h => h.key('Escape'),
    h => { h.document.hidden = true; h.event(h.document, 'visibilitychange'); },
    h => h.event(h.window, 'pagehide'),
    h => { h.context.state.loaded = false; },
  ]) {
    const h = harness(t);
    h.key('KeyW'); await h.advance(200); stop(h);
    const before = h.camera(); await h.advance(1000);
    assert.deepEqual(h.camera(), before); assert.equal(h.frames.size, 0);
    h.document.hidden = false; h.context.state.loaded = true;
    h.key('KeyW', { repeat: true }); await h.advance(200);
    assert.deepEqual(h.camera(), before);
  }
});

test('input descendants, composition, shortcuts and unloaded scenes do not consume WASD', async t => {
  const h = harness(t);
  for (const fields of [{ ctrlKey: true }, { metaKey: true }, { altKey: true }, { isComposing: true }]) {
    assert.equal(h.key('KeyW', fields).defaultPrevented, false);
    await h.advance(100);
  }
  const childEvent = new Event('keydown', { cancelable: true });
  Object.defineProperties(childEvent, { target: { value: {} }, code: { value: 'KeyW' } });
  h.panels.vpWorld.dispatchEvent(childEvent);
  assert.equal(childEvent.defaultPrevented, false);
  h.context.state.loaded = false;
  assert.equal(h.key('KeyW').defaultPrevented, false);
  await h.advance(100);
  assert.deepEqual(h.camera().lookat, [0, 0, 0.5]);
  assert.equal(h.posts.length, 0);
});

test('both panels acquire focus on click and drive the shared camera endpoint', async t => {
  const h = harness(t), recorded = [];
  for (const [panel, code] of [[h.panels.vpWorld, 'KeyW'], [h.panels.vpEgo, 'KeyD']]) {
    h.event(panel, 'pointerdown', { clientX: 0, clientY: 0, pointerId: 1 });
    assert.equal(h.document.activeElement, panel);
    h.event(panel, 'pointerup');
    h.key(code, {}, panel); await h.advance(500); h.release(code);
    await h.step(0);
    const last = h.posts.at(-1);
    assert.deepEqual(last.slice(0, 2), ['POST', '/console/view']);
    recorded.push(JSON.parse(JSON.stringify(last[2], (_, value) => typeof value === 'number' ? +value.toFixed(6) : value)));
  }
  h.event(h.panels.vpEgo, 'pointerdown', { clientX: 0, clientY: 0 });
  h.event(h.panels.vpEgo, 'pointermove', { clientX: 100, clientY: -40 });
  h.event(h.panels.vpEgo, 'pointerup'); await h.step(0);
  recorded.push(JSON.parse(JSON.stringify(h.posts.at(-1)[2], (_, value) => typeof value === 'number' ? +value.toFixed(6) : value)));
  const expected = JSON.parse(readFileSync(new URL('expected/console-navigation.json', import.meta.url), 'utf8'));
  assert.deepEqual(recorded, expected);
  assert.match(html, /id="vpWorld" tabindex="0"/);
  assert.match(html, /id="vpEgo" tabindex="0"/);
});

test('camera posts follow render FPS and flush the final position when released', async t => {
  const h = harness(t, { fps: 5 });
  h.key('KeyW'); await h.advance(1000);
  assert.equal(h.posts.length, 5);
  await h.advance(40); h.release('KeyW'); await h.step(0);
  assert.equal(h.posts.length, 6);
  close(h.posts.at(-1)[2].camera.lookat[1], 2.08);
});

test('slow updates coalesce to the latest position and failure stops movement visibly', async t => {
  let resolvePost, rejectPost;
  const h = harness(t, { api: () => new Promise((resolve, reject) => { resolvePost = resolve; rejectPost = reject; }) });
  h.key('KeyW'); await h.advance(500);
  assert.equal(h.posts.length, 1);
  resolvePost(); await h.step(0);
  assert.equal(h.posts.length, 2);
  close(h.posts[1][2].camera.lookat[1], 1);
  await h.advance(200);
  rejectPost(new Error('offline')); await h.step(0);
  assert.equal(h.frames.size, 0);
  const before = h.camera(); await h.advance(1000);
  assert.deepEqual(h.camera(), before);
  assert.equal(h.posts.length, 2);
  assert.match(h.errors[0], /视角更新失败.*offline.*重新操作/);
});
