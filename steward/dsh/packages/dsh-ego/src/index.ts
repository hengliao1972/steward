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
export type EgoSkill = 'navigate' | 'pick' | 'place' | 'pour' | 'open' | 'close'

/** The top face of an Ego-State object as the head camera sees it now. */
export interface InspectResult {
  entity: string
  /** Grid size: `rows` holds `px` rows of `px` cells. */
  px: number
  /** Heading of the object's frame in the map, degrees. */
  yawDeg: number
  sizeM: number[]
  /** Share of cells seen (not out of view or hidden). */
  coverage: number
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
   * What the robot camera detects right now.
   * @param signal - caller cancellation.
   * @returns visible categories and ranges, plus a saved camera image path when available.
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
   * Look closely at an object's top face (one fresh camera frame).
   * @param entity - Ego-State object id.
   * @param px - grid size, 4 to 64.
   * @param signal - caller cancellation.
   * @returns the sampled face colours in the object's own frame.
   */
  abstract inspect(entity: string, px: number, signal?: AbortSignal): Promise<InspectResult>

  /**
   * Find where objects' top faces appear on a reference object's top face printed at the same
   * scale, from the faces the robot remembers (refreshed first for those in view).
   * @param entities - Ego-State ids of the objects whose faces are sought.
   * @param reference - Ego-State id of the object showing the whole picture.
   * @param px - cells across each object's shorter side, 4 to 16.
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
}
