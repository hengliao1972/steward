/**
 * Service Provider of `ctx.ego` that calls the robot-local Ego API of a Steward Ego Runtime.
 * @module @steward/dsh-ego-runtime-client
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { EgoError, EgoService } from '@steward/dsh-ego'
import type { Annotation, EgoQuery, EgoSkill, FaceMatchResult, InspectResult, JsonObject, PredictionReceipt, ProblemReport, SkillJob } from '@steward/dsh-ego'
import { baseUrl, HttpJsonError, requestJson } from '@steward/http-json'

/** Validated plugin configuration. */
export interface Config {
  /** Base URL of the robot's Ego API. */
  endpoint: string
  /** Deadline for ordinary requests in milliseconds. */
  timeoutMs: number
  /**
   * Control owner sent with skill and explore requests. While a client holds exclusive control of
   * the robot, the Ego Runtime only starts skills for requests naming that owner.
   */
  owner?: string
}

/** `ctx.ego` provider backed by HTTP calls to the Ego API. */
export class EgoRuntimeClient extends EgoService {
  static Config = z.object({
    endpoint: z.string().default('http://127.0.0.1:7420'),
    timeoutMs: z.number().min(100).default(15_000),
    owner: z.string(),
  })

  private readonly base: string

  constructor(ctx: Context, readonly config: Config) {
    super(ctx)
    this.base = baseUrl(config.endpoint)
  }

  private async call<T>(method: 'GET' | 'POST', path: string, body?: unknown, signal?: AbortSignal,
    timeoutMs = this.config.timeoutMs): Promise<T> {
    try {
      return await requestJson<T>(this.base, method, path, body, { signal, timeoutMs })
    } catch (error) {
      if (error instanceof HttpJsonError) throw new EgoError(error.message, error.status)
      throw error
    }
  }

  summary(signal?: AbortSignal): Promise<JsonObject> {
    return this.call('GET', '/ego/summary', undefined, signal)
  }

  async query(query: EgoQuery, signal?: AbortSignal): Promise<JsonObject[]> {
    const params = new URLSearchParams()
    if (query.category) params.set('category', query.category)
    if (query.near) params.set('near', query.near)
    if (query.radius !== undefined) params.set('radius', String(query.radius))
    const result = await this.call<{ objects: JsonObject[] }>('GET', `/ego/query?${params}`, undefined, signal)
    return result.objects
  }

  observe(signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/ego/observe', {}, signal)
  }

  explore(signal?: AbortSignal): Promise<SkillJob> {
    return this.call('POST', '/ego/explore', { owner: this.config.owner }, signal)
  }

  predict(skill: EgoSkill, params: JsonObject, signal?: AbortSignal): Promise<PredictionReceipt> {
    return this.call('POST', '/ego/predict', { skill, params }, signal)
  }

  receipt(id: string, signal?: AbortSignal): Promise<PredictionReceipt> {
    return this.call('GET', `/ego/receipt?id=${encodeURIComponent(id)}`, undefined, signal)
  }

  inspect(entity: string, px: number, signal?: AbortSignal): Promise<InspectResult> {
    return this.call('POST', '/ego/inspect', { entity, px }, signal)
  }

  matchFace(entities: string[], reference: string, px: number, signal?: AbortSignal): Promise<FaceMatchResult> {
    return this.call('POST', '/ego/match-face', { entities, reference, px }, signal)
  }

  annotate(annotation: Annotation, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/ego/annotate', annotation, signal)
  }

  run(receiptId: string, signal?: AbortSignal): Promise<SkillJob> {
    return this.call('POST', '/skill/run', { receipt: receiptId, owner: this.config.owner }, signal)
  }

  job(jobId: string, signal?: AbortSignal): Promise<SkillJob> {
    return this.call('GET', `/skill/status?job=${encodeURIComponent(jobId)}`, undefined, signal)
  }

  wait(jobId: string, timeoutS: number, signal?: AbortSignal): Promise<SkillJob> {
    return this.call('POST', '/skill/wait', { job: jobId, timeoutS }, signal, timeoutS * 1000 + this.config.timeoutMs)
  }

  stop(signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/skill/stop', {}, signal)
  }

  reportProblem(report: ProblemReport, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/ego/alert', { ...report, source: 'agent', level: 'blocked' }, signal)
  }
}

export default EgoRuntimeClient
