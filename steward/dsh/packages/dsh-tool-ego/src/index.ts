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
import { recordingId } from '@steward/dsh-ego'
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
  for (const key of ['drive', 'correctOnce', 'method', 'selectedHoleIds', 'stageScrews', 'closeRelease', 'recoverDropped', 'wingRetry', 'part', 'screw', 'body', 'stages', 'stagedPrediction', 'target', 'object', 'support', 'spot', 'dock', 'reach', 'lift', 'pathLength', 'wristYaw', 'roll']) {
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
  return { job: job.id, skill: job.skill, state: job.state, message: job.message, timing: job.timing, log: job.log.slice(-8),
    ...(job.postcondition === undefined ? {} : { postcondition: job.postcondition }) }
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
        + 'positions are estimates. orientation explains whether yawDeg is only a box axis: front/back and near-square '
        + 'axes may be ambiguous. screwHoles includes source and frame. Outline-compatible priors retain source=prior; '
        + 'positionsByRegistration lists alternative entity-frame positions (one shared index for all holes on an object). '
        + 'orientation.holeRegistration explains compatibility and unresolved half turns. Unregistered templates cannot '
        + 'drive alignment: reobserve dimensions/features or use ordinary placement. A label alone is not registration. '
        + 'appearance says what you remember of an object\'s top face: how long ago you '
        + 'saw it, the share seen, plain / interiorColors (plain true or interiorColors under about 20: a cardboard '
        + 'back; do not use spread alone — lighting makes a plain back look varied) and stale (put down since, look '
        + 'again); ego_match_face compares remembered faces.',
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
      description: 'Report what your available cameras detect right now (categories, distances, which camera, and screwHoles on '
        + 'parts that show mating holes). Your world model also updates continuously; use ego_state for the '
        + 'accumulated picture, including screwHoles once a detection has been fused.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await ego.observe(exec.signal))
      },
    }),
    defineTool({
      name: 'ego_capture_cameras',
      description: 'Capture your left, original and right RGB cameras at one sensor timestamp without moving. '
        + 'Saves three JPEG images and a manifest in your workspace and returns their paths, pixel intrinsics K, '
        + 'and fixed camera_to_reference calibration in OpenCV axes, metres, relative to the original head camera. '
        + 'Includes gripper_pose from the same encoder timestamp, with gripper_to_camera transforms for all three views. '
        + 'Works independently of the console camera display mode. No absolute camera poses or synthesized views are provided.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await ego.captureCameras(exec.signal))
      },
    }),
    defineTool({
      name: 'ego_rgb_recording_start',
      description: 'Start background RGB recording at a simulation-clock FPS. Records only head_camera. '
        + 'with same-timestamp measured odometry, encoders and calibration. Does not move the robot. Start before issuing navigation '
        + 'with skill_run wait=false; let this recorder sample independently of model/tool latency, then stop after navigation completes. '
        + 'Do not approximate 2 FPS by repeatedly calling ego_capture_cameras. Recording stops at max_frames or fails visibly on missed samples.',
      parameters: { fps: { type: 'number', description: 'Simulation-time FPS, default 2.' },
        max_frames: { type: 'number', description: 'Recording frame bound, default 256; provider bounds apply.' },
        request_key: { type: 'string', required: true, description: 'Stable 8–128 character recording-start key; reuse after lost responses.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      execute: async (args, exec) => json(await ego.startRgbRecording(args.fps ?? 2, args.max_frames ?? 256, args.request_key, exec.signal)),
    }),
    defineTool({
      name: 'ego_rgb_recording_status', description: 'Inspect recording status, actual sensor counts, simulation timestamps and errors. Does not command motion.',
      parameters: { recording_id: { type: 'string', required: true, description: 'Actual rgb_ recording identity.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      execute: async (args, exec) => json(await ego.rgbRecordingStatus(recordingId(args.recording_id), exec.signal)),
    }),
    defineTool({
      name: 'ego_rgb_recording_stop', description: 'Stop sampling and finalize all queued RGB evidence from the selected recording cameras; '
        + 'cancel=true explicitly discards the recording. If status is finalizing, query status until terminal. Does not stop or start robot motion.',
      parameters: { recording_id: { type: 'string', required: true, description: 'Actual recording identity.' },
        cancel: { type: 'boolean', description: 'Default false: preserve evidence. True cancels recording.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      execute: async (args, exec) => json(await ego.stopRgbRecording(recordingId(args.recording_id), args.cancel ?? false, exec.signal)),
    }),
    defineTool({
      name: 'ego_camera_calibration',
      description: 'Read fixed RGB camera intrinsics and camera_to_reference transforms into the middle head_camera frame '
        + '(OpenCV x right, y down, z forward; translation in metres) without images or motion. '
        + 'Each camera has pixel K and dimensions. No absolute camera pose is provided.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await ego.cameraCalibration(exec.signal))
      },
    }),
    defineTool({
      name: 'ego_gripper_pose',
      description: 'Read your gripper-link pose relative to your base and the robot cameras from current joint encoders, '
        + 'your body model and fixed camera calibration. No motion or image capture. gripper_to_camera maps gripper-local '
        + 'points (x forward, y left, z up) into each OpenCV camera frame (x right, y down, z forward), translations in metres. '
        + 'The gripper frame is the gripper link origin. Includes sensor timestamp, measured lift/reach and gripper_to_base. '
        + 'For a pose synchronized to images, use ego_capture_cameras and its embedded gripper_pose instead of combining separate calls.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await ego.gripperPose(exec.signal))
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
        + 'motion. For a component that must be fastened, prefer "fasten_onto"; use "assemble_onto" for placement only onto a stationary supported body: params {part: ego id, body: ego id, hole?: complete body feature prefix, heldHole?: complete component prefix}. '
        + 'Start empty-handed. The skill actively obtains close views aimed at the body with independent camera baselines, then freezes the registered body frame. If baselines are insufficient it can use checked single/two-view geometry at low confidence; registrationAttempts reports sample rejection reasons. It picks only the loose part while retaining its pre-grasp heading, fits the complete interfaces, places, retreats and reobserves all released components. '
        + 'Call predict then skill_run with the receipt and wait for the terminal job before calling again. Reuse the original body id while that body stays stationary. When a released component becomes the next supporting body, use postcondition.releasedReference.entity: its raised frame is propagated from registered grasp geometry and the executed release, explicitly low confidence with assumed settling height. A lost/merged mask is not a new observation. '
        + 'The staged receipt is admission to the controller: each motion has its own fresh risk check; later stages can fail. Do not repeat a part if postcondition.releaseExecuted is true, even if retreat failed. '
        + 'postcondition reports registrationEvidence (source, confidence, views), samples with relative poses and planarPassed, plus stableObserved; passed=null means full verification is unresolved, especially directed yaw or uprightness. Release execution is not assembly success or fastening. '
        + 'orientation.stationaryAssembly exposes frozen registration time, raw observations, released parts and invalidation. Missing/merged detections do not invalidate a stationary frame; a body/support grasp or clear motion does. '
        + 'Keep the body stationary during each registered placement and do not pick installed components. A wider part can rest on a narrow body when its centre is supported; this does not certify static stability. Use the manual to select the supporting side and complete interface groups, compare relative heights and top/underside features after settling, and report unknown orientation or fastening honestly. Follow the operator/manual task order. Complete required body/tool pose preparation before freezing a frame; a later required pose change invalidates it and requires fresh observations and registration. Never relabel a merged object to bypass refusal. '
        + 'Other skills: "navigate" (needs x, y, optional yaw in radians), "pick" (needs target; a pourable '
        + 'container is picked with what lies in it once nothing lies on top of it), "place" (needs on: a surface, a '
        + 'part that has screw holes to mate onto, an open drawer to put the object into, or a larger bowl to nest a '
        + 'bowl in; optional offset; or at: [x, y] to put it down on the floor; when the held part and the mate both '
        + 'have screw holes, place seats those holes (on the mate itself, or on a surface with alignTo naming the mate; '
        + 'optional hole and heldHole select observed or outline-registered feature ids or prefixes; ambiguous fits list '
        + 'candidate world yaw (radians) and offset in the on-surface frame, ranked against the pre-transport estimated heading when available. '
        + 'assemble_onto and place perform placement only. fasten_onto {part,body,hole?,heldHole?,drive?} adds deterministic screw seating, optional driving and perceptual verification. The default drive=false selects seating; request drive=true when powered driving is required; it does not read or change environment mode. Follow the task-required component order, tool orientation and body orientation. '
        + 'Attempt fixation and inspect independent evidence. Missing optional measurements produce warnings and catalogue/grasp estimates; continue feasible stages and report unverified attachment separately from action execution. '
        + 'seat_screw {part,body,hole,method?} defaults to ordinary pick/place at the just-released component holes, propagated from grasp-relative geometry and release encoders, then checks independent screw geometry. fasten_onto batches the ordinary releases before a shared view. Failed or unknown landings share at most two replays per job; this posterior recovery is independent of correctOnce and wingRetry; method="contact" selects the previous endpoint controller; hole must be one exact body hole id. regrasp_upright {target,on?} stands any elongated tool on a perceived support and regrips it from above. drive_screw {screw} (or {body,hole}) grips a perceived powered driver, turns its estimated tip down, contacts the best screw-head estimate and holds. Missing endpoints use catalogue geometry transported from grasp with encoders. '
        + 'flip_assembly {body,turnOver?} lifts, checks independent following, places the load and registers its best estimated face. turnOver defaults to true; false performs a lift/set-down following check. Loose components without confirmed screw landings produce an advisory warning before flipping; the pinch targets the body outside their volumes. fasten_onto continues through the first screw attempts, but fails if any requested landing remains failed or unknown after recovery. Read postcondition.confirmedSeatedScrews and requestedScrews; a confirmed landing is independent visual placement evidence, not mechanical attachment. Each starts with an empty hand. '
        + 'hole/heldHole accept actual feature ids or shared id prefixes from screwHoles; the group metadata string is not itself a selector. '
        + 'For fitting a multi-hole interface use its complete group/prefix; only seat_screw subsequently selects a single aligned hole. Do not use a single pair to bypass a failed fit: one pair cannot constrain rotation. '
        + 'Register each part independently: box yaw is the measured box axis, not a shared design heading. Derive the relative axes from the hole '
        + 'templates and outline registration; do not copy the body yaw or add a guessed quarter turn. Record the clean pre-pick frame before any '
        + 'parking or regrasp. Grasp heading ranks equivalent fits and cancels transport; it does not prove the initial heading was correct. '
        + 'Ties use stable template/feature order. After release, retract and move to a clear, different view; require fresh observations of BOTH parts '
        + 'after releasedAt, not the commanded spot or a merged family box. Compare the relative centre and visible features with each template hypothesis, '
        + 'including measurement uncertainty. Correct only when new evidence rejects the current hypothesis and supports another: re-identify the loose part, '
        + 'pick it, predict with the supported candidate yaw/offset, release and remeasure. If a half turn leaves the outline and features identical, '
        + 'relative centres cannot distinguish it: retain unknown directed yaw and do not alternate blindly. A noncentral feature on the mate does not '
        + 'make a symmetric held part directional. During planar mating keep the registered body stable and supported, moving only the loose component. '
        + 'Within the task-required order, keep the next interface visible and reachable. Required changes of body pose are separate operations: '
        + 'use observed support and orientation, reobserve after the change, and discard the old registration. Do not balance it on a loose component or treat a combined mask as evidence of attachment. '
        + 'Check the release height and approach clearance as well as XY: alignTo does not plan insertion underneath a body, and a table-height '
        + 'fallback at its centre may collide with it. Side/underside interfaces need observed geometry in their new pose; the planar skill cannot validate arbitrary 3D insertion. Report an unsupported step rather than guessing. '
        + 'Check top/side features, vertical extent and support before grasping and releasing; avoid wrist roll/flip during planar mating. '
        + 'orientation.posture reports a shape alarm, not a measured upright angle: unknown or dimensionsCompatible does not prove uprightness. '
        + 'Read orientation.stationaryAssembly after assemble_onto: status=frozen retains the pre-addition frame independently of merged masks. For low-level place, orientation.assembly contains envelope corroboration only. '
        + 'Do not re-register holes against the combined outer axis or relabel the body to bypass a refusal. envelope_observed only corroborates the envelope; '
        + 'geometry_observed corroborates a separate component box; neither verifies directed yaw, uprightness or fastening. If that evidence is unresolved, '
        + 'obtain a new clear view. If the body appears tipped or a change is unexplained by the ledger, stop mating. Park the loose part on a separate clear stable surface. '
        + 'Recover the body only when independent side/top observations establish its tilt and a supported recovery motion is feasible; otherwise '
        + 'report posture unresolved. Do not use a guessed roll or alignHoles:false on the suspect body to bypass the refusal. '
        + 'A planar fit does not establish seating or attachment. '
        + 'Keep a ledger of released parts, their source positions, occupied interfaces and verification evidence. Do not pick an aligned, released and verified '
        + 'part again. Exclude all installed parts from loose-part selection, including after track renumbering. Before moving a body or support, explicitly '
        + 'list the completed relations that this would invalidate and a plan to reverify them; a merged perception label is not a mechanical attachment. '
        + 'After every addition, wait for the running skill to become terminal, retract and observe the stationary body and released component. Complete and verify any task-required fixation before selecting the next loose part. '
        + 'Use a complete compatible prefix on each side, predict again from the current state, and run that feasible receipt without adding a guessed yaw. '
        + 'After every addition, recheck earlier relations and the stationary body. Keep released-unverified, verified and invalidated entries distinct. '
        + 'Before declaring completion, observe all required relations together after settling from more than one clear view; do not proceed to unrelated '
        + 'work once the requested subset is independently verified. A succeeded release alone never completes a relation. '

        + 'Catalogue group/mates data restrict default interface pairing; explicit hole/heldHole override it. Supply yaw to choose a candidate; '
        + 'if several offsets remain, also supply offset. Base the choice on camera evidence and verify after release; alignHoles: false keeps a manual '
        + 'offset). Ordinary planar placement of an interface part also restores its grasp heading when yaw is omitted; explicit yaw overrides it. '
        + 'Check the predicted spot against the intended feature: ordinary placement can move an occupied spot and provides no precision mating guarantee; '
        + 'optional roll in radians tips it about the arm before release, e.g. π/2 = 90° '
        + 'to set a flat object down on its side; not together with flip), "pour" (the held object must be pourable; needs on: a surface; tips it over the '
        + 'spot so its contents slide out there in a heap, then holds it upright again, still in the gripper; the spot '
        + 'keeps the contents\' slide from every edge so nothing falls off; count the contents before pouring, since '
        + 'flat ones can land wholly hidden under one another), "open" / '
        + '"close" (needs target: a drawer; the robot pulls or pushes its handle and ends standing back so the inside '
        + 'is in view). Objects lying on the floor, such as chairs or a dropped flat piece, are picked from '
        + 'beside them; look at one first from where it is in view (ego_inspect\'s hint says how far back), since '
        + 'docked beside it the cameras no longer see it. Small ones are lifted clear and carried like objects taken '
        + 'off a table; large ones are carried a few centimetres above the floor in front of the robot, and put down '
        + 'with at; '
        + 'while carrying them the robot needs more room to turn. Objects inside a drawer are only seen while it is '
        + 'open. If the risk is too high or the skill is infeasible, follow the issues (for example move closer and '
        + 'observe, or choose another spot) and predict again. An infeasible receipt lists blockers: entities in the '
        + 'way, whether they can be moved, and parking spots (parkAt) out of the way; clear a movable blocker first '
        + '(pick it, place it at a parking spot), then predict the original skill again. A receipt can be run once '
        + 'and expires. For a yaw-controlled placement, postcondition.orientation reports the independently '
        + 'observed outer-axis error after release; null directedDeltaDeg means front/back is still unknown. '
        + 'If unknown, obtain a fresh unobstructed camera view and reread ego_state placementEvidence. '
        + 'A correction proposal keeps the measured centre: re-identify the object, predict pick, then predict '
        + 'the supplied place parameters and reobserve. Confirm asymmetric features before accepting a directed orientation.',
      parameters: {
        skill: { type: 'string', enum: ['regrasp_upright', 'assemble_onto', 'fasten_onto', 'seat_screw', 'drive_screw', 'flip_assembly', 'navigate', 'pick', 'place', 'pour', 'open', 'close'], required: true, description: 'Skill to predict.' },
        turnOver: { type: 'boolean', description: 'flip_assembly: turn the assembly over (default true); false lifts and sets down to inspect following.' },
        drive: { type: 'boolean', description: 'fasten_onto: false seats screws without using a powered driver; default false; explicitly set true to drive. Action choice only, independent of environment mode.' },
        correctOnce: { type: 'boolean', description: 'Simple seating: observe and compensate a fresh out-of-capture landing error (default true).' },
        stageScrews: { type: 'boolean', description: 'Optional pre-staging; default false. Adds extra transfers before component placement.' },
        closeRelease: { type: 'boolean', description: 'Optional smaller estimated release gap; default false.' },
        recoverDropped: { type: 'boolean', description: 'Optional bounded re-placement after an observed drop; default false.' },
        wingRetry: { type: 'boolean', description: 'Optional changed-station component retry and extra measured fastener correction; default false.' },
        method: { type: 'string', enum: ['simple', 'contact'], description: 'seat_screw/drive_screw/fasten_onto: simple ordinary manipulation (default), or legacy contact controller.' },
        screw: { type: 'string', description: 'drive_screw: robot-perceived screw id returned by seat_screw.' },
        part: { type: 'string', description: 'assemble_onto/fasten_onto: perceived loose component id; seat_screw: the already aligned, released component id.' },
        body: { type: 'string', description: 'assembly/fastening/flip_assembly: supported body id; for a newly released body use releasedReference.entity and inspect its low-confidence frame.' },
        target: { type: 'string', description: 'pick: id of the object to grasp; open/close: id of the drawer.' },
        on: { type: 'string', description: 'place: id of the surface, screw-hole part, open drawer or larger bowl to put the held object on or in; pour: id of the surface.' },
        offset: { type: 'array', items: { type: 'number' }, description: 'place, pour: [dx, dy] of the spot from the surface centre in the surface frame, metres; without it the free spot nearest you. If something already lies at the requested spot, the free spot nearest it is used instead and the issues say where (the receipt plan.spot holds the spot used). During hole alignment, selects a fitted candidate offset within sensor tolerance; incompatible offsets are rejected, never substituted for the hole fit.' },
        alignTo: { type: 'string', description: 'place: id of the part whose screw holes the held part should seat on. Use when on is a table or other surface and the mate is another object on it; omit when on itself is the mate.' },
        hole: { type: 'string', description: 'place/assemble_onto/fasten_onto: mate hole id or complete prefix; seat_screw/drive_screw: one exact body hole id. Overrides catalogue interface pairing; ambiguous geometry still returns a default plan and alternatives for post-release verification.' },
        heldHole: { type: 'string', description: 'place/assemble_onto: observed or registered prior held-object hole id or prefix to use; with no selectors, catalogue-compatible complete groups are tried; without group data all held holes must fit. Explicit selection overrides catalogue pairing.' },
        alignHoles: { type: 'boolean', description: 'place: seat screw holes when both parts have them (default true when on is the mate or alignTo is set). Set false to keep a manual offset.' },
        at: { type: 'array', items: { type: 'number' }, description: 'place on the floor: [x, y] in metres.' },
        x: { type: 'number', description: 'navigate: goal x in metres.' },
        y: { type: 'number', description: 'navigate: goal y in metres.' },
        yaw: { type: 'number', description: 'navigate: goal heading; place: optional absolute world yaw of the held object\'s estimated frame (as ego_state yawDeg reports it), in radians, not a relative wrist turn. A single hole does not constrain yaw: omitting it restores the estimated world heading recorded at grasp, cancelling transport rotation (falls back to the current held heading if no grasp snapshot exists). This preserves initial orientation uncertainty; supply a visually grounded yaw to change orientation. For circular or near-square objects the frame may have no observable physical direction, so this command cannot certify a semantic front. Verify asymmetric features after release. If template holes are unregistered, yaw alone does not register them; use observed features or explicit ordinary placement with alignHoles=false and offset.' },
        flip: { type: 'boolean', description: 'place: turn the held object upside down before setting it down. pick: take it so that it can be turned over later; some arms take it tilted for that, and a place with flip says when the object must be picked again with flip.' },
        roll: { type: 'number', description: 'place: tip the held object about the arm before setting it down, radians (π/2 = 90° onto its side). Not together with flip.' },
        clearance: { type: 'number', description: 'place, pour: least gap in metres kept to objects already on the surface (default 0.015). With an offset, 0 allows contact or overlap (interlocking pieces, stacking): the spot is used exactly as requested, never shifted aside. Screw-hole alignment always uses 0.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        const params: JsonObject = {}
        // Tool default is seating; powered driving is an explicit action choice.
        if (args.skill === 'fasten_onto' && args.drive === undefined) params.drive = false
        for (const key of ['drive', 'correctOnce', 'method', 'stageScrews', 'closeRelease', 'recoverDropped', 'wingRetry', 'part', 'screw', 'body', 'turnOver', 'target', 'on', 'offset', 'alignTo', 'hole', 'heldHole', 'alignHoles', 'at', 'x', 'y', 'yaw', 'flip', 'roll', 'clearance'] as const) {
          if (args[key] !== undefined) params[key] = args[key]
        }
        return json(receiptView(await ego.predict(args.skill, params, exec.signal)))
      },
    }),
    defineTool({
      name: 'ego_inspect',
      description: 'Look closely at the top face of an object with one fresh frame of each camera, read from the camera that sees most of it: its colours on a px x px '
        + 'grid. rows is in the object\'s own frame (row 0 is its +y edge, column 0 its -x edge, as seen from above '
        + 'with its heading yawDeg). worldRows is the same face as seen from above with the map\'s +y up and +x to the '
        + 'right: rows turned worldTurnDeg counter-clockwise (yawDeg rounded to a quarter turn), so faces of objects '
        + 'lying at different headings compare directly in worldRows. Putting an object down with place yaw = its '
        + 'current yawDeg + k x 90 degrees (given in radians) turns its face k quarter turns counter-clockwise as seen '
        + 'from above. Each cell is [r, g, b] or null where the face is out of view or hidden. Stand within about '
        + 'a metre with the object in front of you. You have two cameras on your head: head_camera looks ahead and sees a '
        + 'surface only from some way ahead of it; wide_camera, a short-focal-length camera tilted further down, also sees '
        + 'the strip of a table next to the edge you stand at and the gripper at work, in less detail. When part of the face is unseen, hint says '
        + 'why (too close below the view, beside or behind it, too far, hidden) and how to move. Use it to compare '
        + 'printed or patterned faces, read which side of a flat object is up, or tell similar objects apart. A '
        + 'printed flat object lying with its plain back up: use plain === true or interiorColors (under about 20). '
        + 'Do not use spread / interiorSpread alone — table lighting makes cardboard look variegated in RGB even when '
        + 'it is one colour. Turn it over (place with flip) and look again. Flat objects lying on or against each other hide '
        + 'parts of their faces and can be perceived as one object, and one lying on another can hide it wholly: set '
        + 'them apart on free space first, looking again where each one lay after picking it up. The object frame stays '
        + 'fixed to the object between views and while it is carried, so one inspection before picking it tells which '
        + 'place yaw turns its face as you want.',
      parameters: {
        entity: { type: 'string', required: true, description: 'Id of the object to inspect.' },
        px: { type: 'number', description: 'Grid size, 4 to 128 (default 24). Raise it when sourcePx is large and you need pattern detail.' },
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
        + 'ambiguous, so look closer (raise px toward 32) and match again. offsetM is good to about one cell (cellM): '
        + 'when you know the layout (for example a grid of equal pieces), snap it to the nearest slot. '
        + 'plain / interiorColors (under about 20) means a cardboard back facing up; do not trust spread alone. '
        + 'An entry with error says what to look at.',
      parameters: {
        entities: { type: 'array', items: { type: 'string' }, required: true, description: 'Ids of the objects whose faces are sought.' },
        reference: { type: 'string', required: true, description: 'Id of the object showing the whole picture.' },
        px: { type: 'number', description: 'Cells across each object\'s shorter side, 4 to 32 (default 24).' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await ego.matchFace(args.entities as string[], args.reference, args.px ?? 24, exec.signal))
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
      name: 'ego_forget',
      description: 'Remove an object from the Ego State as a false perception. Use it only for an object that is '
        + 'physically implausible (marked `unsupported`: floating with nothing under it, or inside another solid object) '
        + 'and that you looked for again, with ego_observe from a place where its spot is in view, without perceiving it '
        + '(its lastSeenAgoS keeps growing). The robot refuses an object it holds or perceived in the last few seconds.',
      parameters: {
        entity: { type: 'string', required: true, description: 'Id of the object.' },
        reason: { type: 'string', required: true, description: 'Why it is implausible and how you looked for it again.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await ego.forget({ entity: args.entity, reason: args.reason }, exec.signal))
      },
    }),
    defineTool({
      name: 'ego_candidates',
      description: 'Find the objects that could be what the instruction calls something when perception names it differently: '
        + 'the segmenter names things by appearance (a lid printed with a picture can be "picture"), and the classifier only '
        + 'knows the shapes of its categories. Ranks objects by how well their perceived category alternatives, segmenter '
        + 'names and size fit `name` and `size`; without either, lists the objects whose current name is doubtful. Each '
        + 'carries `doubts` and the evidence behind its name. Confirm with ego_inspect or a closer look, then ego_annotate.',
      parameters: {
        name: { type: 'string', description: 'What you are looking for, as a category word, for example "box" or "lid".' },
        size: { type: 'array', items: { type: 'number' }, description: 'One to three approximate dimensions in metres from the instruction or your expectation.' },
        exclude: { type: 'array', items: { type: 'string' }, description: 'Object ids to leave out.' },
        limit: { type: 'number', description: 'Maximum number of candidates.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json({ candidates: await ego.candidates({ name: args.name, size: args.size, exclude: args.exclude, limit: args.limit }, exec.signal) })
      },
    }),
    defineTool({
      name: 'skill_run',
      description: 'Execute the plan of a receipt from ego_predict. Waits for the skill to finish unless wait is '
        + 'false. Staged assembly skills can take tens of minutes: a wait timeout leaves the job running. '
        + 'Poll its status; keep each run_code wait at most 240 seconds and do not body_stop just because a poll times out. '
        + 'While such a skill runs the robot may leave the table, change viewpoints and fetch parts on its own; a status that stays running or docked for many minutes is normal. '
        + 'Interrupting a running assembly skill usually leaves the part placed but not fastened. When it ends, confirm with ego_state that the part is on its body before the next step. '
        + 'If one sub-goal shows no real progress after three high-level attempts, switch to another sub-goal and return to it later. '
        + 'The robot refuses receipts that are infeasible, expired, already used, or above the risk limit.',
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
