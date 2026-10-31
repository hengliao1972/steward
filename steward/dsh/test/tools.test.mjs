import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, readdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { egoTools } from '@steward/dsh-tool-ego'
import { CapabilityRunLedger, CapabilityStore, capabilityTools, taskSpecifics } from '@steward/dsh-tool-ego-capabilities'
import { worldTools } from '@steward/dsh-tool-world'
import { receiptDenial, reviewExecution } from '@steward/dsh-robot-safety'

function fakeEgo(receipts = {}) {
  const calls = []
  const job = (state) => ({ id: 'job-1', skill: 'pick', receipt: 'rcpt-1', state, message: '', log: [] })
  return {
    calls,
    summary: async () => ({ objects: [] }),
    query: async (q) => { calls.push(['query', q]); return [] },
    observe: async () => ({ visible: [] }),
    explore: async () => job('running'),
    predict: async (skill, params) => {
      calls.push(['predict', skill, params])
      return { id: 'rcpt-1', skill, params, risk: 0.1, feasible: true, withinRiskLimit: true, maxRisk: 0.35, issues: [],
        expiresAt: 60, plan: { target: params.target, path: [[0, 0], [1, 1]], reach: 0.3 } }
    },
    receipt: async (id) => {
      if (!(id in receipts)) throw new Error(`unknown receipt '${id}'`)
      return receipts[id]
    },
    run: async (id) => { calls.push(['run', id]); return job('running') },
    job: async () => job('succeeded'),
    wait: async (id, timeoutS) => { calls.push(['wait', id, timeoutS]); return job('succeeded') },
    stop: async () => ({ stopped: null }),
    reportProblem: async (report) => { calls.push(['report', report]); return { id: 1, ...report } },
    inspect: async (entity, px) => { calls.push(['inspect', entity, px]); return { entity, px, rows: [], coverage: 0 } },
    matchFace: async (entities, reference, px) => { calls.push(['matchFace', entities, reference, px]); return { reference, results: [] } },
    annotate: async (a) => { calls.push(['annotate', a]); return { id: a.entity, category: a.label } },
  }
}

const exec = (name, args) => ({ name, arguments: args, signal: new AbortController().signal, callId: 'c1' })

function tool(tools, name) {
  const found = tools.find(t => t.name === name)
  assert.ok(found, `tool ${name} is registered`)
  return found
}

test('ego tools expose the robot vocabulary without World State concepts', () => {
  const names = egoTools(fakeEgo()).map(t => t.name)
  assert.deepEqual(names, ['ego_state', 'ego_query', 'ego_observe', 'ego_explore', 'ego_predict', 'ego_inspect',
    'ego_match_face', 'ego_annotate', 'skill_run', 'skill_status', 'body_stop', 'report_problem'])
  for (const t of egoTools(fakeEgo())) assert.doesNotMatch(t.description, /ground truth|world state/i)
})

test('ego_predict forwards only the given parameters and hides the raw path', async () => {
  const ego = fakeEgo()
  const result = await tool(egoTools(ego), 'ego_predict').execute({ skill: 'pick', target: 'e:4' }, exec('ego_predict'))
  assert.deepEqual(ego.calls[0], ['predict', 'pick', { target: 'e:4' }])
  assert.equal(result.receipt, 'rcpt-1')
  assert.equal(result.plan.path, undefined)
  assert.equal(result.plan.reach, 0.3)
})

test('skill_run waits for completion by default', async () => {
  const ego = fakeEgo()
  const result = await tool(egoTools(ego), 'skill_run').execute({ receipt: 'rcpt-1' }, exec('skill_run'))
  assert.deepEqual(ego.calls.map(c => c[0]), ['run', 'wait'])
  assert.equal(result.state, 'succeeded')
})

test('world_edit maps floor and support placements to World Control API edits', async () => {
  const edits = []
  const world = { edit: async (request) => { edits.push(request); return { ok: true, ...request } } }
  const edit = tool(worldTools(world), 'world_edit')
  await edit.execute({ action: 'move', entity: 'w:cup-1', on: 'w:sideboard', offset_x: 0.1 }, exec('world_edit'))
  await edit.execute({ action: 'spawn', entity: 'w:bowl-2', asset: 'bowl', x: 1, y: -0.5, yaw: 0.3 }, exec('world_edit'))
  await edit.execute({ action: 'remove', entity: 'w:plate-1' }, exec('world_edit'))
  assert.deepEqual(edits, [
    { action: 'move', entity: 'w:cup-1', on: 'w:sideboard', offset: [0.1, 0] },
    { action: 'spawn', entity: 'w:bowl-2', asset: 'bowl', at: { x: 1, y: -0.5 }, yaw: 0.3 },
    { action: 'remove', entity: 'w:plate-1' },
  ])
  await assert.rejects(edit.execute({ action: 'move', entity: 'w:cup-1', x: 1 }, exec('world_edit')), /both x and y/)
  await assert.rejects(edit.execute({ action: 'spawn', entity: 'w:x', asset: 'cup' }, exec('world_edit')), /needs a placement/)
})

test('report_problem forwards the goal, step, diagnosis and message to the operator alert', async () => {
  const ego = fakeEgo()
  const report = { goal: '把红碗放到操作台', step: '抓取 e:4', diagnosis: '抓取两次都没对准：e:4 位置不确定度 4 cm', message: '请检查碗的位置' }
  const result = await tool(egoTools(ego), 'report_problem').execute({ ...report, extra: 'dropped' }, exec('report_problem'))
  assert.deepEqual(ego.calls.at(-1), ['report', report])
  assert.equal(result.diagnosis, report.diagnosis)
})

test('safety policy denies skill_run without a receipt, with an unknown one, or above the risk limit', async () => {
  const receipts = {
    ok: { id: 'ok', risk: 0.1, feasible: true, issues: [] },
    risky: { id: 'risky', risk: 0.6, feasible: true, issues: ['target position uncertainty 4.0 cm'] },
    used: { id: 'used', risk: 0.1, feasible: true, issues: [], used: true },
  }
  const ego = fakeEgo(receipts)
  assert.equal((await reviewExecution(ego, exec('skill_run', {}), 0.35)).kind, 'deny')
  assert.match((await reviewExecution(ego, exec('skill_run', { receipt: 'nope' }), 0.35)).reason, /could not be verified/)
  assert.match((await reviewExecution(ego, exec('skill_run', { receipt: 'risky' }), 0.35)).reason, /above the limit/)
  assert.match((await reviewExecution(ego, exec('skill_run', { receipt: 'used' }), 0.35)).reason, /already executed/)
  assert.equal(await reviewExecution(ego, exec('skill_run', { receipt: 'ok' }), 0.35), undefined)
  assert.equal(await reviewExecution(ego, exec('ego_state', {}), 0.35), undefined)
  assert.equal(receiptDenial(receipts.ok, 0.05), 'receipt ok has risk 0.1 above the limit 0.05: ')
})

test('ego_predict passes a placing heading, a flip and a clearance; pour is a skill', async () => {
  const ego = fakeEgo()
  const predict = tool(egoTools(ego), 'ego_predict')
  await predict.execute({ skill: 'place', on: 'e:4', offset: [0.1, 0], yaw: 1.5, flip: true, clearance: 0 }, exec('ego_predict'))
  await predict.execute({ skill: 'pour', on: 'e:4' }, exec('ego_predict'))
  assert.deepEqual(ego.calls, [
    ['predict', 'place', { on: 'e:4', offset: [0.1, 0], yaw: 1.5, flip: true, clearance: 0 }],
    ['predict', 'pour', { on: 'e:4' }],
  ])
})

test('ego_inspect and ego_annotate reach the Ego service', async () => {
  const ego = fakeEgo()
  const tools = egoTools(ego)
  await tool(tools, 'ego_inspect').execute({ entity: 'e:7' }, exec('ego_inspect'))
  await tool(tools, 'ego_match_face').execute({ entities: ['e:7', 'e:8'], reference: 'e:1' }, exec('ego_match_face'))
  await tool(tools, 'ego_annotate').execute({ entity: 'e:7', label: 'box', affordances: ['graspable', 'pourable'] }, exec('ego_annotate'))
  assert.deepEqual(ego.calls, [['inspect', 'e:7', 24], ['matchFace', ['e:7', 'e:8'], 'e:1', 12], ['annotate', { entity: 'e:7', label: 'box', affordances: ['graspable', 'pourable'] }]])
})

test('a saved capability is listed without its code, fetched with it, and survives a new store', async () => {
  const root = join(await mkdtemp(join(tmpdir(), 'caps-')), 'library')
  const tools = capabilityTools(new CapabilityStore(root))
  assert.deepEqual(await tool(tools, 'capability_list').execute({}, exec('capability_list')), { capabilities: [] })
  await tool(tools, 'capability_save').execute({ name: 'sort_tiles', layer: 'skill', description: 'Sort tiles by colour.',
    requires: 'tiles on a surface', ensures: 'tiles sorted', parameters: { tiles: 'ids' }, code: 'return args.tiles.length' },
  exec('capability_save'))
  const again = capabilityTools(new CapabilityStore(root))
  const listed = await tool(again, 'capability_list').execute({}, exec('capability_list'))
  assert.equal(listed.capabilities.length, 1)
  assert.equal(listed.capabilities[0].code, undefined, 'the list stays short')
  assert.equal(listed.capabilities[0].status, 'candidate')
  const cap = await tool(again, 'capability_get').execute({ name: 'sort_tiles' }, exec('capability_get'))
  const run = new (Object.getPrototypeOf(async function () {}).constructor)('tools', 'args', cap.code)
  assert.equal(await run({}, { tiles: ['e:1', 'e:2'] }), 2, 'the code runs as the tool description shows')
  assert.equal(cap.uses, 1)
  assert.deepEqual(await readdir(root), ['sort_tiles.json'])
  await assert.rejects(tool(again, 'capability_get').execute({ name: 'missing' }, exec('capability_get')), /no capability/)
  await assert.rejects(tool(again, 'capability_save').execute({ name: '../x', layer: 'skill', description: 'd', requires: '',
    ensures: '', code: '' }, exec('capability_save')), /capability names/)
})

test('code naming objects of one task or holding many decimal constants is not saved', async () => {
  const store = new CapabilityStore(join(await mkdtemp(join(tmpdir(), 'caps-')), 'library'))
  const base = { layer: 'skill', description: 'd', requires: '', ensures: '', parameters: {} }
  await assert.rejects(store.save({ ...base, name: 'grab', code: 'await tools.skill_run({ target: "e:42" })' }), /e:42/)
  const many = 'return [0.115, 0.52, -0.69, 0.71, 0.33, 0.25, 0.47]'
  await assert.rejects(store.save({ ...base, name: 'grid', code: many }), /7 decimal constants/)
  assert.deepEqual(taskSpecifics('const pitch = args.size + 0.01; return pitch * 0.5', 6), [])
})

test('runs of a fetched capability are counted from the program that fetched it', async () => {
  const store = new CapabilityStore(join(await mkdtemp(join(tmpdir(), 'caps-')), 'library'),
    { verifyAfterRuns: 2, deprecateAfterFailures: 2, maxDecimalLiterals: 6 })
  const base = { layer: 'skill', description: 'd', requires: '', ensures: '', parameters: {}, code: 'return 1' }
  await store.save({ ...base, name: 'flip' })
  const fetch = (root) => [{ callId: `${root}:1`, rootCallId: root, name: 'capability_get', arguments: { name: 'flip' } }, { isError: false }]
  const program = (root, ok) => [{ callId: root, name: 'run_code',
    arguments: { sessionId: 'session-1', turn: Number(root.slice(1)) } },
    ok ? { isError: false } : { isError: true, error: { message: 'grasp missed' } }]
  const first = new CapabilityRunLedger(store, 'run-1')
  await first.observe(...fetch('t1'))
  assert.equal((await first.observe(...program('t1', true)))[0].status, 'candidate', 'one run is not yet proof')
  await first.observe(...fetch('t2'))
  assert.equal((await first.observe(...program('t2', true)))[0].status, 'verified', 'two turns of one task count as two runs')
  assert.deepEqual(await first.observe(...program('t3', true)), [], 'a program that fetched nothing counts for nothing')
  const second = new CapabilityRunLedger(store, 'run-2')
  for (const root of ['t4', 't5']) {
    await second.observe(...fetch(root))
    await second.observe(...program(root, false))
  }
  const failing = await store.read('flip')
  assert.equal(failing.status, 'deprecated')
  assert.equal(failing.runs.last.error, 'grasp missed')
  const changed = await store.save({ ...base, name: 'flip', code: 'return 2' })
  assert.deepEqual([changed.version, changed.status, changed.runs.succeeded], [2, 'candidate', 0], 'new code is a new candidate')
})

test('a forgotten capability leaves the list and keeps its reason', async () => {
  const root = join(await mkdtemp(join(tmpdir(), 'caps-')), 'library')
  const tools = capabilityTools(new CapabilityStore(root))
  await tool(tools, 'capability_save').execute({ name: 'old_way', layer: 'recipe', description: 'd', requires: '',
    ensures: '', code: 'return 1' }, exec('capability_save'))
  await tool(tools, 'capability_delete').execute({ name: 'old_way', reason: 'superseded by new_way' }, exec('capability_delete'))
  assert.deepEqual(await tool(tools, 'capability_list').execute({}, exec('capability_list')), { capabilities: [] })
  await assert.rejects(tool(tools, 'capability_get').execute({ name: 'old_way' }, exec('capability_get')), /no capability/)
  const [file] = await readdir(join(root, 'forgotten'))
  assert.match(file, /^old_way-/)
})
