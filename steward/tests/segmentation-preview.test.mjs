/** Preview polling exercises the shipped console handlers without running SAM. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { test } from 'node:test';

const html = readFileSync(new URL('../src/steward/world/console.html', import.meta.url), 'utf8');
const script = html.slice(html.indexOf('const samShown ='), html.indexOf('// ------------------------------------------------------------------ scenarios & bodies'));

test('completed frames, unchanged frames, failures and phase changes', async () => {
  const elements = new Map();
  const element = () => ({ hidden: true, textContent: '', children: [], style: {},
    replaceChildren(...children) { this.children = children; this.textContent = ''; },
    removeAttribute(name) { delete this[name]; },
  });
  let result = { active: true, error: null, frame: { id: 'first', stamp: 12.5,
    image: 'data:image/jpeg;base64,test', labels: [{ id: 1, category: 'cup', score: 0.9, color: '#00ff00' }] } };
  let failed = false;
  const requests = [];
  const context = vm.createContext({
    robotId: 'robot', ROBOT_CAMERAS: ['head_camera'], URLSearchParams,
    document: { createElement: element },
    $: id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    setText: (node, text) => { node.textContent = text; },
    fetch: async url => { requests.push(url); if (failed) throw new Error('offline');
      return { ok: true, json: async () => result }; },
  });
  vm.runInContext(script, context);
  const refresh = () => vm.runInContext('refreshSegmentation()', context);
  await refresh();
  const image = elements.get('samImage-head_camera');
  const caption = elements.get('samCaption-head_camera');
  assert.equal(image.hidden, false);
  assert.match(caption.textContent, /12.50/);
  assert.match(elements.get('samLabels-head_camera').children[0].textContent, /cup 90%/);
  const { image: ignored, ...unchanged } = result.frame;
  result = { active: true, error: 'SAM unreachable', frame: unchanged };
  await refresh();
  assert.match(requests.at(-1), /after=first/);
  assert.equal(image.src, 'data:image/jpeg;base64,test');
  assert.match(caption.textContent, /SAM unreachable/);
  failed = true;
  await refresh();
  assert.match(caption.textContent, /offline/);
  assert.equal(image.hidden, false);
  failed = false;
  result = { active: false, error: null, frame: null };
  await refresh();
  assert.equal(image.hidden, true);
  assert.equal(image.src, undefined);
  assert.match(caption.textContent, /phase1-sam/);
  assert.equal(elements.get('samLabels-head_camera').children.length, 0);
});

test('console script parses', () => {
  new vm.Script(html.split('<script>')[1].split('</script>')[0]);
});
