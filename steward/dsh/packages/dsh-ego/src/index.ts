/**
 * Service Definition of the robot Ego capability, mounted as `ctx.ego`.
 *
 * The Ego capability is the Ego Harness view of one robot: the Ego-State (the robot's own,
 * partial and uncertain reconstruction of the world), ESim predictions that return Prediction
 * Receipts, and execution of exactly the plan a receipt predicted. Providers talk to the
 * robot's Ego Runtime; no provider may expose World State.
 * @module @steward/dsh-ego
 */

import { Service, type Context } from '@deepseek-ai/cordis'

import { brandString, type Branded } from '@deepseek-ai/dsh-brand'

/** Robot-owned RGB recording identity. */
export type RecordingId = Branded<'StewardRgbRecordingId'>
/** Admit a recording identity from model/tool JSON or a sensor response.
 * @param value - remote recording identity.
 * @returns validated recording identity.
 */
export function recordingId(value: string): RecordingId {
  if (!/^rgb_[a-f0-9]{32}$/.test(value)) throw new EgoError('Invalid RGB recording identity')
  return brandString<RecordingId>(value)
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    ego: EgoService
  }
}

/** JSON object as returned by the Ego Runtime. */
export interface JsonObject {
  [key: string]: unknown
}

/** Skills the Ego Runtime can predict and run. */
export type EgoSkill = 'navigate' | 'pick' | 'place' | 'pour' | 'open' | 'close' | 'drive_screw' | 'insert_screw_into_hole'

/** The top face of an Ego-State object as the robot camera that sees most of it shows it now. */
export interface InspectResult {
  entity: string
  /** Grid size: `rows` holds `px` rows of `px` cells. */
  px: number
  /** Sensor id of the camera the face was read from (`head_camera` or `wide_camera`). */
  camera: string
  /** Heading of the object's frame in the map, degrees. */
  yawDeg: number
  sizeM: number[]
  /** Share of cells seen (not out of view or hidden). */
  coverage: number
  /** Colour standard deviation of seen cells. Lighting inflates this on a plain back; prefer `plain` / `interiorColors`. */
  spread?: number
  /** Colour standard deviation excluding a one-cell border (neighbours often tint the edge). */
  interiorSpread?: number
  /** Share of interior cells seen. */
  interiorCoverage?: number
  /** Distinct interior colours after 16-level quantisation; a plain cardboard back stays near 1–4, a print has dozens. */
  interiorColors?: number
  /** True when `interiorColors` is low enough that the face is treated as a uniform back. */
  plain?: boolean
  /** Present when part of the face is unseen: why, from each camera's view (too close below it, beside or behind it, too far, hidden). */
  hint?: string
  /** Row 0 is the object's +y edge, column 0 its -x edge; each cell is [r, g, b] (0-255) or null. */
  rows: (number[] | null)[][]
  /** `rows` turned `worldTurnDeg` counter-clockwise: seen from above, row 0 faces the map's +y and columns run towards +x. */
  worldRows: (number[] | null)[][]
  /** The heading `yawDeg` rounded to a quarter turn (0, 90, 180 or 270): how far `worldRows` is turned from `rows`. */
  worldTurnDeg: number
  stamp: number
}

/** One placement of an object's face on a reference face (see {@link EgoService.matchFace}). */
export interface FaceMatch {
  /** Where the face centre lies on the reference, in the reference frame from its centre, metres. */
  offsetM: number[]
  /** Quarter turns counter-clockwise (seen from above) that align the face with the reference, degrees. */
  turnDeg: number
  /** The object heading that aligns its face with the reference as the reference lies now, degrees. */
  placeYawDeg: number
  /** Colour correlation, 1 = identical. */
  score: number
}

/** Where one object's face appears on the reference, or why it could not be matched. */
export interface FaceMatchEntry {
  entity: string
  /** Set instead of the other fields when the object's face is not known well enough. */
  error?: string
  /** Cell size the faces were compared at, metres. */
  cellM?: number
  /** Colour standard deviation of the object's face; near 0 the face is plain. */
  spread?: number
  /** Colour standard deviation excluding a one-cell border; near 0 means a plain back. */
  interiorSpread?: number
  /** Share of the object's face the robot has seen. */
  faceCoverage?: number
  /** World seconds since the robot last saw the object's face. */
  seenAgoS?: number
  /** Best placements first, each at least half a face apart from the better ones. */
  matches?: FaceMatch[]
}

/** Where objects' faces appear on a reference face printed at the same scale. */
export interface FaceMatchResult {
  reference: string
  /** Share of the reference face the robot has seen. */
  referenceCoverage: number
  results: FaceMatchEntry[]
  stamp: number
}

/** The robot agent's identification of an Ego-State object. */
export interface Annotation {
  entity: string
  /** Category name to use from now on, for example `box`. */
  label?: string
  /** Affordances replacing the category prior, for example `graspable`, `pourable`, `support-surface`. */
  affordances?: string[]
}

/** The robot agent's judgement that an Ego-State object is a false perception. */
export interface Forgetting {
  entity: string
  /** What made the object implausible and how its place was looked at again without perceiving it. */
  reason: string
}

/** What the instruction asked for, to find the objects that could be it (see {@link EgoService.candidates}). */
export interface CandidateQuery {
  /** Category, role or segmenter word sought, for example `box`. */
  name?: string
  /** One to three approximate dimensions in metres, any order. */
  size?: number[]
  /** Object ids to leave out. */
  exclude?: string[]
  /** Maximum number of candidates; the runtime default applies when absent. */
  limit?: number
}

/** An object with the evidence behind its name, for comparison with an instruction. */
export interface Candidate {
  id: string
  category: string
  role: string | null
  extentM: number[] | null
  color: string | null
  on: string | null
  position: number[]
  lastSeenAgoS: number
  /** Category probabilities from perception alone, ignoring any label. */
  perceived?: Record<string, number>
  /** Mean segmenter score per category, with the prompt words that map to it. */
  segmenter?: Record<string, { score: number; prompts: string[] }>
  /** The label the robot agent gave it, if any. */
  label?: string
  /** Reasons to doubt its current name; empty when the name is well supported. */
  doubts: string[]
  /** Fit to the sought name and size; present only for a query with either. */
  score?: number
  why?: string[]
}

/** A failed robot step to reflect on (see {@link EgoService.reflect}). */
export interface FailureReport {
  /** Tool that failed. */
  tool?: string
  /** Its arguments; ids among them (`target`, `on`, `params.target`, ...) name the objects involved. */
  arguments?: JsonObject
  /** The failure text: error, receipt issues or skill message. */
  message: string
  /** What the instruction asked for, when known. */
  name?: string
  size?: number[]
}

/** Evidence for reflecting on a failed step. */
export interface Reflection {
  /** The kind of failure and what usually resolves it. */
  cause: { key: string; label: string; advice: string }
  /** The name and size the candidates were ranked against. */
  sought: { name: string | null; size: number[] | null }
  /** The objects the failed call was about. */
  targets: Candidate[]
  /** Ids the call named that the Ego-State no longer holds. */
  missing: string[]
  /** Other objects that could be the sought one, then doubtfully named ones. */
  candidates: Candidate[]
}

/** Filter for Ego-State object queries. */
export interface EgoQuery {
  /** Category to keep, for example `cup`. */
  category?: string
  /** Ego entity id whose neighbourhood to search. */
  near?: string
  /** Neighbourhood radius in metres; the runtime default applies when absent. */
  radius?: number
}

/** ESim result that authorises running one predicted plan once before it expires. */
/** An entity in the way of a predicted skill. */
export interface ReceiptBlocker {
  id: string
  category: string
  /** True for floor-standing graspable entities the robot can carry out of the way. */
  movable: boolean
  /** Free floor spots [x, y] where a movable blocker can be put down. */
  parkAt: number[][]
}

export interface PredictionReceipt {
  id: string
  skill: EgoSkill
  params: JsonObject
  /** Risk score in [0, 1] derived from Ego-State uncertainty and geometry. */
  risk: number
  feasible: boolean
  /** True when feasible and the risk does not exceed the runtime limit. */
  withinRiskLimit: boolean
  maxRisk: number
  issues: string[]
  /** Robot clock time after which the receipt is rejected. */
  expiresAt: number
  /** Robot clock when the receipt was read; used with `expiresAt` / `expired`. */
  now?: number
  /** True when the robot clock has passed `expiresAt`. */
  expired?: boolean
  plan: JsonObject
  /** Entities that made an infeasible prediction fail; movable ones come with parking spots. */
  blockers?: ReceiptBlocker[]
  used?: boolean
}

/** One skill execution. */
/** A problem the robot cannot resolve by itself, as the operator console shows it. */
export interface ProblemReport {
  /** What the robot is trying to achieve. */
  goal: string
  /** The step it was carrying out when it got stuck. */
  step: string
  /** Its own analysis of its tool log: the dominant cause and the evidence for it. */
  diagnosis: string
  /** What happened, what was tried, and what the operator needs to do. */
  message: string
}

export interface SkillJob {
  id: string
  skill: string
  receipt: string
  state: 'running' | 'succeeded' | 'failed' | 'stopped'
  message: string
  log: string[]
}

/** Failure reported by an Ego provider, carrying the runtime status when one exists. */
export class EgoError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
    this.name = 'EgoError'
  }
}

/**
 * The `ego` service. Every method accepts the caller's cancellation signal; providers must
 * stop waiting when it aborts.
 */
export abstract class EgoService extends Service {
  constructor(ctx: Context) {
    super(ctx, 'ego')
  }

  /**
   * Robot pose, held object, current skill and the objects in the Ego-State.
   * @param signal - caller cancellation.
   * @returns the compact Ego-State summary.
   */
  abstract summary(signal?: AbortSignal): Promise<JsonObject>

  /**
   * Objects in the Ego-State matching a filter.
   * @param query - category and neighbourhood filter.
   * @param signal - caller cancellation.
   * @returns matching objects with position, uncertainty and support.
   */
  abstract query(query: EgoQuery, signal?: AbortSignal): Promise<JsonObject[]>

  /**
   * What the robot cameras detect right now.
   * @param signal - caller cancellation.
   * @returns visible categories, ranges and the camera of each, plus saved camera image paths when available.
   */
  abstract observe(signal?: AbortSignal): Promise<JsonObject>

  /**
   * Start looking around in place to extend the Ego-State.
   * @param signal - caller cancellation.
   * @returns the started job.
   */
  abstract explore(signal?: AbortSignal): Promise<SkillJob>

  /**
   * Ask ESim to predict a skill and obtain a receipt.
   * @param skill - skill to predict.
   * @param params - skill parameters (target, support, goal pose).
   * @param signal - caller cancellation.
   * @returns the Prediction Receipt.
   */
  abstract predict(skill: EgoSkill, params: JsonObject, signal?: AbortSignal): Promise<PredictionReceipt>

  /**
   * Read an issued receipt.
   * @param id - receipt id.
   * @param signal - caller cancellation.
   * @returns the receipt as the runtime stores it.
   */
  abstract receipt(id: string, signal?: AbortSignal): Promise<PredictionReceipt>

  /**
   * Look closely at an object's top face (one fresh frame of each camera).
   * @param entity - Ego-State object id.
   * @param px - grid size, 4 to 128.
   * @param signal - caller cancellation.
   * @returns the sampled face colours in the object's own frame.
   */
  abstract inspect(entity: string, px: number, signal?: AbortSignal): Promise<InspectResult>

  /**
   * Locate screw holes on an Ego-State object in the world frame and whether each faces up.
   * @param entity - Ego-State object id.
   * @param signal - caller cancellation.
   * @returns hole centres/normals in world metres, facing (`up`/`side`/`down`), and `holesFaceUp`.
   */
  abstract screwHoles(entity: string, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Find where objects' top faces appear on a reference object's top face printed at the same
   * scale, from the faces the robot remembers (refreshed first for those in view).
   * @param entities - Ego-State ids of the objects whose faces are sought.
   * @param reference - Ego-State id of the object showing the whole picture.
   * @param px - cells across each object's shorter side, 4 to 32.
   * @param signal - caller cancellation.
   * @returns the best placements of each face on the reference.
   */
  abstract matchFace(entities: string[], reference: string, px: number, signal?: AbortSignal): Promise<FaceMatchResult>

  /**
   * Record what the robot agent has identified an object to be; ESim and the estimator use the
   * label and affordances in place of the classifier's category prior.
   * @param annotation - object id, label and affordances.
   * @param signal - caller cancellation.
   * @returns the object's Ego-State summary entry after the change.
   */
  abstract annotate(annotation: Annotation, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Remove an object the robot agent judged a false perception. The robot refuses while the object
   * is held or was perceived within its `forget_unseen_s` (3 s by default).
   * @param forgetting - object id and the reason.
   * @param signal - caller cancellation.
   * @returns the removal record as listed in the Ego-State summary's `dropped`.
   */
  abstract forget(forgetting: Forgetting, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Objects that could be what a name and approximate size refer to, best first, each with its
   * naming evidence and doubts; without a name or size, the doubtfully named objects.
   * @param query - sought name and size.
   * @param signal - caller cancellation.
   * @returns the candidates.
   */
  abstract candidates(query: CandidateQuery, signal?: AbortSignal): Promise<Candidate[]>

  /**
   * Evidence for reflecting on a failed step: its cause, the objects involved and candidates for
   * the object the instruction meant.
   * @param failure - the failed call and its failure text.
   * @param signal - caller cancellation.
   * @returns the reflection evidence.
   */
  abstract reflect(failure: FailureReport, signal?: AbortSignal): Promise<Reflection>

  /**
   * Execute the plan of a receipt. The runtime re-checks validity, expiry and risk.
   * @param receiptId - receipt to execute.
   * @param signal - caller cancellation.
   * @returns the started job.
   */
  abstract run(receiptId: string, signal?: AbortSignal): Promise<SkillJob>

  /**
   * Read a job.
   * @param jobId - job id.
   * @param signal - caller cancellation.
   * @returns the job record.
   */
  abstract job(jobId: string, signal?: AbortSignal): Promise<SkillJob>

  /**
   * Wait until a job leaves the running state or the timeout passes.
   * @param jobId - job id.
   * @param timeoutS - maximum wait in seconds.
   * @param signal - caller cancellation.
   * @returns the job record at return time.
   */
  abstract wait(jobId: string, timeoutS: number, signal?: AbortSignal): Promise<SkillJob>

  /**
   * Stop all motion and hold the current pose.
   * @param signal - caller cancellation.
   * @returns the id of the interrupted job, if any.
   */
  abstract stop(signal?: AbortSignal): Promise<JsonObject>

  /**
   * Tell the operator about a situation the robot cannot resolve by itself; the Steward console
   * shows it as an alert until the operator acknowledges it.
   * @param report - the goal, the step, the robot's own diagnosis, and what it needs.
   * @param signal - caller cancellation.
   * @returns the recorded alert.
   */
  abstract reportProblem(report: ProblemReport, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Save synchronized images from the robot's existing cameras.
   * @param signal - caller cancellation.
   * @returns shared sensor timestamp, saved image paths, camera calibration and the same-sample gripper pose; no absolute poses.
   */
  abstract captureCameras(signal?: AbortSignal): Promise<JsonObject>

  /**
   * Read fixed RGB intrinsics and relative extrinsics without imaging or motion.
   * @param signal - caller cancellation.
   * @returns RGB calibration; camera_to_reference maps each OpenCV camera into the middle camera, in metres.
   */
  abstract cameraCalibration(signal?: AbortSignal): Promise<JsonObject>

  /** Start an independent RGB recorder; it never commands motion.
   * @param fps - requested sensor cadence in simulation time.
   * @param maxFrames - hard frame bound, at least two.
   * @param requestKey - stable start identity, reused after uncertain responses.
   * @param signal - local cancellation; an accepted recording continues.
   * @returns recording identity and status.
   */
  abstract startRgbRecording(fps: number, maxFrames: number, requestKey: string, signal?: AbortSignal): Promise<JsonObject>

  /** Inspect a middle RGB recording.
   * @param id - actual recording identity.
   * @param signal - local cancellation.
   * @returns cadence, sensor counts and terminal errors.
   */
  abstract rgbRecordingStatus(id: RecordingId, signal?: AbortSignal): Promise<JsonObject>

  /** Stop sampling and finalize evidence, or explicitly discard a cancelled recording.
   * @param id - recording identity.
   * @param cancel - false preserves evidence; true cancels recording.
   * @param signal - local cancellation, not recording cancellation.
   * @returns recording status; finalizing requires another status query.
   */
  abstract stopRgbRecording(id: RecordingId, cancel: boolean, signal?: AbortSignal): Promise<JsonObject>

  /** Export all middle-camera RGB samples, measured poses and current gripper geometry.
   * @param id - successfully completed recording.
   * @param signal - local cancellation.
   * @returns inline sensor evidence for local preparation; providers must strip image bytes before model output.
   */
  abstract rgbRecordingEvidence(id: RecordingId, signal?: AbortSignal): Promise<JsonObject>

  /** Export a finished sensor recording as a NovelAnything scene, without inference or motion.
   * @param id - successful middle RGB recording.
   * @param sourceCount - explicit 2–16 context count; undefined selects up to 16 evenly spaced frames.
   * @param signal - local cancellation.
   * @returns ZIP bytes encoded for the local provider and metadata; callers must save and strip archive_base64 before model output.
   */
  abstract prepareNovelAnything(id: RecordingId, sourceCount: number | undefined, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Read gripper-link pose from joint encoders and the robot body model without imaging or motion.
   * @param signal - caller cancellation.
   * @returns sensor timestamp, joint readings and transforms from gripper axes (forward/left/up) to base and OpenCV camera frames, in metres.
   */
  abstract gripperPose(signal?: AbortSignal): Promise<JsonObject>
}
