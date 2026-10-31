/**
 * Consumer of `ctx.ego`: the robot tools the Ego Harness model calls.
 *
 * The model sees the world only through the robot's Ego-State. Every motion that can collide
 * or drop something follows predict → run: `ego_predict` asks ESim for a Prediction Receipt and
 * `skill_run` executes exactly that receipt's plan. Looking around in place needs no receipt.
 * @module @steward/dsh-tool-ego
 */

import type { Context } from '@deepseek-ai/cordis'
import { defineTool, type InferValue, type ToolDefinition } from '@deepseek-ai/dsh-tools'
import type { EgoService, JsonObject, PredictionReceipt, SkillJob } from '@steward/dsh-ego'

export const name = 'steward-tool-ego'
export const inject = ['tools', 'ego']

const JSON_OUTPUT = { type: 'json' } as const
type Json = InferValue<typeof JSON_OUTPUT>

/** Lossless JSON copy of a runtime response, typed as a tool value. */
function json(value: unknown): Json {
  return JSON.parse(JSON.stringify(value)) as Json
}

function text(value: unknown): [{ type: 'text'; text: string }] {
  return [{ type: 'text', text: JSON.stringify(value, null, 1) }]
}

function receiptView(r: PredictionReceipt): JsonObject {
  const plan = r.plan as Record<string, unknown>
  const brief: JsonObject = {}
  for (const key of ['target', 'object', 'support', 'spot', 'dock', 'reach', 'lift', 'pathLength', 'wristYaw', 'roll']) {
    if (key in plan) brief[key] = plan[key]
  }
  const view: JsonObject = {
    receipt: r.id, skill: r.skill, feasible: r.feasible, risk: r.risk, maxRisk: r.maxRisk,
    withinRiskLimit: r.withinRiskLimit, issues: r.issues, expiresAt: r.expiresAt, plan: brief,
  }
  if (r.blockers && r.blockers.length) view.blockers = r.blockers.map(b => ({ ...b }))
  return view
}

function jobView(job: SkillJob): JsonObject {
  return { job: job.id, skill: job.skill, state: job.state, message: job.message, log: job.log.slice(-8) }
}

/**
 * Build the robot tool definitions over an Ego service.
 * @param ego - the `ctx.ego` service.
 * @returns tool definitions in registration order.
 */
export function egoTools(ego: EgoService): ToolDefinition[] {
  return [
    defineTool({
      name: 'ego_state',
      description: 'Read your current understanding of the world: your own position, what you are holding, the skill '
        + 'in progress, and every object you have perceived with its estimated position, position uncertainty in '
        + 'metres, size, what it rests on, and when you last saw it. Objects you have not seen are absent, and '
        + 'positions are estimates. appearance says what you remember of an object\'s top face: how long ago you '
        + 'saw it, the share seen, its colour spread (near 0: a plain face) and stale (put down since, look again); '
        + 'ego_match_face compares remembered faces.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await ego.summary(exec.signal))
      },
    }),
    defineTool({
      name: 'ego_query',
      description: 'Find perceived objects by category (for example "cup", "table") and/or near another object. '
        + 'Matching uses each object\'s role (the scene-level reading of what it is) and then its category '
        + 'probabilities, best match first, so a sideboard partly seen and labelled "counter" is still found.',
      parameters: {
        category: { type: 'string', description: 'Object category or role to look for.' },
        near: { type: 'string', description: 'Id of an object whose surroundings to search, for example "e:3".' },
        radius: { type: 'number', description: 'Search radius in metres around `near` (default 1).' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json({ objects: await ego.query({ category: args.category, near: args.near, radius: args.radius }, exec.signal) })
      },
    }),
    defineTool({
      name: 'ego_observe',
      description: 'Report what your camera detects right now (categories and distances). Your world model also '
        + 'updates continuously; use ego_state for the accumulated picture.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await ego.observe(exec.signal))
      },
    }),
    defineTool({
      name: 'ego_explore',
      description: 'Turn in place through a full circle to perceive your surroundings. Use it first in a new place '
        + 'or when an object you need is not in ego_state.',
      parameters: {
        timeoutS: { type: 'number', description: 'Maximum seconds to wait for completion (default 120).' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        const job = await ego.explore(exec.signal)
        return json(jobView(await ego.wait(job.id, args.timeoutS ?? 120, exec.signal)))
      },
    }),
    defineTool({
      name: 'ego_predict',
      description: 'Predict the consequences of a skill before doing it. Returns a receipt with a risk score, the '
        + 'issues found (for example a target whose position is too uncertain to grasp reliably), and the planned '
        + 'motion. Skills: "navigate" (needs x, y, optional yaw in radians), "pick" (needs target; a pourable '
        + 'container is picked with what lies in it once nothing lies on top of it), "place" (needs on: a surface, an '
        + 'open drawer to put the object into, or a larger bowl to nest a bowl in; optional offset; or at: [x, y] to '
        + 'put it down on the floor), "pour" (the held object must be pourable; needs on: a surface; tips it over the '
        + 'spot so its contents slide out there in a heap, then holds it upright again, still in the gripper; the spot '
        + 'keeps the contents\' slide from every edge so nothing falls off; count the contents before pouring, since '
        + 'flat ones can land wholly hidden under one another), "open" / '
        + '"close" (needs target: a drawer; the robot pulls or pushes its handle and ends standing back so the inside '
        + 'is in view). Floor-standing objects such as chairs are picked from '
        + 'beside them, carried a few centimetres above the floor in front of the robot, and put down with at; '
        + 'while carrying them the robot needs more room to turn. Objects inside a drawer are only seen while it is '
        + 'open. If the risk is too high or the skill is infeasible, follow the issues (for example move closer and '
        + 'observe, or choose another spot) and predict again. An infeasible receipt lists blockers: entities in the '
        + 'way, whether they can be moved, and parking spots (parkAt) out of the way; clear a movable blocker first '
        + '(pick it, place it at a parking spot), then predict the original skill again. A receipt can be run once '
        + 'and expires.',
      parameters: {
        skill: { type: 'string', enum: ['navigate', 'pick', 'place', 'pour', 'open', 'close'], required: true, description: 'Skill to predict.' },
        target: { type: 'string', description: 'pick: id of the object to grasp; open/close: id of the drawer.' },
        on: { type: 'string', description: 'place: id of the surface, open drawer or larger bowl to put the held object on or in; pour: id of the surface.' },
        offset: { type: 'array', items: { type: 'number' }, description: 'place, pour: [dx, dy] of the spot from the surface centre in the surface frame, metres; without it the free spot nearest you. If something already lies at the requested spot, the free spot nearest it is used instead and the issues say where (the receipt plan.spot holds the spot used).' },
        at: { type: 'array', items: { type: 'number' }, description: 'place on the floor: [x, y] in metres.' },
        x: { type: 'number', description: 'navigate: goal x in metres.' },
        y: { type: 'number', description: 'navigate: goal y in metres.' },
        yaw: { type: 'number', description: 'navigate: goal heading; place: final heading of the object\'s frame (as ego_state yawDeg and ego_inspect report it), radians; the wrist turns it.' },
        flip: { type: 'boolean', description: 'place: turn the held object upside down before setting it down.' },
        clearance: { type: 'number', description: 'place, pour: least gap in metres kept to objects already on the surface (default 0.015). With an offset, 0 allows contact or overlap (interlocking pieces, stacking): the spot is used exactly as requested, never shifted aside.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        const params: JsonObject = {}
        for (const key of ['target', 'on', 'offset', 'at', 'x', 'y', 'yaw', 'flip', 'clearance'] as const) {
          if (args[key] !== undefined) params[key] = args[key]
        }
        return json(receiptView(await ego.predict(args.skill, params, exec.signal)))
      },
    }),
    defineTool({
      name: 'ego_inspect',
      description: 'Look closely at the top face of an object with one fresh camera frame: its colours on a px x px '
        + 'grid. rows is in the object\'s own frame (row 0 is its +y edge, column 0 its -x edge, as seen from above '
        + 'with its heading yawDeg). worldRows is the same face as seen from above with the map\'s +y up and +x to the '
        + 'right: rows turned worldTurnDeg counter-clockwise (yawDeg rounded to a quarter turn), so faces of objects '
        + 'lying at different headings compare directly in worldRows. Putting an object down with place yaw = its '
        + 'current yawDeg + k x 90 degrees (given in radians) turns its face k quarter turns counter-clockwise as seen '
        + 'from above. Each cell is [r, g, b] or null where the face is out of view or hidden. Stand within about '
        + 'a metre with the object in front of you; low coverage means move closer and look again. Use it to compare '
        + 'printed or patterned faces, read which side of a flat object is up, or tell similar objects apart. A '
        + 'printed flat object lying with its plain back up shows an almost uniform face: turn it over (place with '
        + 'flip) and look again. Flat objects lying on or against each other hide parts of their faces and can be '
        + 'perceived as one object, and one lying on another can hide it wholly: set them apart on free space first, '
        + 'looking again where each one lay after picking it up. The object frame stays fixed to the object between '
        + 'views and while it is carried, so one inspection before picking it tells which place yaw turns its face '
        + 'as you want.',
      parameters: {
        entity: { type: 'string', required: true, description: 'Id of the object to inspect.' },
        px: { type: 'number', description: 'Grid size, 4 to 64 (default 24).' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await ego.inspect(args.entity, args.px ?? 24, exec.signal))
      },
    }),
    defineTool({
      name: 'ego_match_face',
      description: 'Find where the top faces of objects appear on the top face of a reference object printed at '
        + 'the same scale (puzzle pieces on the picture of their box, tiles on a pattern sheet). It works from the '
        + 'faces you remember: the robot remembers the top face of every object up to about 0.6 m that it sees '
        + 'within 1.5 m (ego_state appearance says how much and how long ago), and refreshes the ones in view '
        + 'first; a face put down since it was last seen must be looked at again. Match all objects of a task in '
        + 'one call. For each it compares the colours of every placement and quarter turn and returns the best '
        + 'few: offsetM is where the face centre belongs on the reference, in the reference frame from its centre; '
        + 'turnDeg is how far the face must turn counter-clockwise to line up with the reference; placeYawDeg is '
        + 'the heading (degrees; place takes radians) that lines it up with the reference as the reference lies '
        + 'now. To build the picture elsewhere at heading H, put each object at centre + offsetM turned by H, with '
        + 'place yaw = H + turnDeg. score is a correlation (1 = identical); a best score barely above the second is '
        + 'ambiguous, so look closer and match again. offsetM is good to about one cell (cellM): when you know the '
        + 'layout (for example a grid of equal pieces), snap it to the nearest slot. spread near 0 (under about 10) '
        + 'means a plain face: the object probably lies with its back up. An entry with error says what to look at.',
      parameters: {
        entities: { type: 'array', items: { type: 'string' }, required: true, description: 'Ids of the objects whose faces are sought.' },
        reference: { type: 'string', required: true, description: 'Id of the object showing the whole picture.' },
        px: { type: 'number', description: 'Cells across each object\'s shorter side, 4 to 16 (default 12).' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await ego.matchFace(args.entities as string[], args.reference, args.px ?? 12, exec.signal))
      },
    }),
    defineTool({
      name: 'ego_annotate',
      description: 'Record what you have identified an object to be when the perceived category is wrong or unknown '
        + '(for example "unknown" for a box or a puzzle piece). Your label replaces its category, and the affordances '
        + 'you name replace the category\'s defaults, so prediction and planning treat it accordingly: graspable (can be '
        + 'picked), support-surface (things can be put on it), receptacle (things can be put in it), container, '
        + 'pourable (can be carried with its contents and poured), flat.',
      parameters: {
        entity: { type: 'string', required: true, description: 'Id of the object.' },
        label: { type: 'string', description: 'Category name to use for it.' },
        affordances: { type: 'array', items: { type: 'string' }, description: 'Its affordances, replacing the defaults.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await ego.annotate({ entity: args.entity, label: args.label, affordances: args.affordances }, exec.signal))
      },
    }),
    defineTool({
      name: 'skill_run',
      description: 'Execute the plan of a receipt from ego_predict. Waits for the skill to finish unless wait is '
        + 'false. The robot refuses receipts that are infeasible, expired, already used, or above the risk limit.',
      parameters: {
        receipt: { type: 'string', required: true, description: 'Receipt id returned by ego_predict.' },
        wait: { type: 'boolean', description: 'Wait for completion (default true).' },
        timeoutS: { type: 'number', description: 'Maximum seconds to wait (default 180).' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        const job = await ego.run(args.receipt, exec.signal)
        if (args.wait === false) return json(jobView(job))
        return json(jobView(await ego.wait(job.id, args.timeoutS ?? 180, exec.signal)))
      },
    }),
    defineTool({
      name: 'skill_status',
      description: 'Read the state of a running or finished skill.',
      parameters: { job: { type: 'string', required: true, description: 'Job id from skill_run.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(jobView(await ego.job(args.job, exec.signal)))
      },
    }),
    defineTool({
      name: 'body_stop',
      description: 'Stop all motion immediately and hold the current pose.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await ego.stop(exec.signal))
      },
    }),
    defineTool({
      name: 'report_problem',
      description: 'Alert the operator that you hit a situation you cannot resolve by yourself: an unexpected '
        + 'error, a skill that keeps failing, or a task that cannot be done from here. Before calling, review your own '
        + 'tool calls in this turn and diagnose: which prediction or skill failed, the error text, how often it '
        + 'repeated, where you were, and how it relates to blockers, uncertainty, or missing or duplicate objects in '
        + 'the ego state. The operator console shows the report as an alert. Call it once, then stop.',
      parameters: {
        goal: { type: 'string', required: true, description: 'What you are trying to achieve.' },
        step: { type: 'string', required: true, description: 'The step you were carrying out when you got stuck.' },
        diagnosis: { type: 'string', required: true, description: 'The cause you diagnosed from your tool log and the evidence for it.' },
        message: { type: 'string', required: true, description: 'What happened, what you tried, and what you need from the operator.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        const { goal, step, diagnosis, message } = args
        return json(await ego.reportProblem({ goal, step, diagnosis, message }, exec.signal))
      },
    }),
  ]
}

export function apply(ctx: Context): void {
  for (const tool of egoTools(ctx.ego)) ctx.tools.register(tool)
}
