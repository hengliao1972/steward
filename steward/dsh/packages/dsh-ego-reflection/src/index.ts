/**
 * Ego Harness reflection on robot steps, and the thinking effort of each step.
 *
 * In the physical world a step usually fails because the scene differs from what the model
 * assumed, not because the program was wrong: perception names objects by shape and appearance
 * (the box of an instruction can be in the Ego-State as `picture`), objects lie hidden, turned or
 * out of place, and a retry of the same call fails the same way. Reflection gives the model the
 * facts (the operator's messages, the failure, earlier failures under the same message, the Ego
 * Runtime's cause and naming evidence from `ctx.ego.reflect`) and then asks generic questions so
 * that the model applies its own knowledge of the physical world. The questions name dimensions
 * to examine, never a remedy for a task. Reflection has three depths:
 *
 * - shallow: a new kind of failure; a few questions on the phenomenon, its likely causes, the
 *   cheapest check and the next step.
 * - deep: a failure that recurs, a failure after `deepAfter` shallow reflections, or a difficulty
 *   the model reports with the `reflect` tool. It adds a digest of the Ego-State scene and asks
 *   macro questions first (restate scene, task, subtasks and failures; how the instruction's
 *   words map onto the perceived objects; which assumptions were never checked), then the
 *   physical causes, then the strategy and its efficiency.
 * - checkpoint: `checkpointEvery` top-level steps without any failure or reflection, or
 *   `idleEvery` top-level steps in a row in which the robot did not act (no `actionTools` call:
 *   only observing, computing or printing); a review of progress, or of what the idle steps
 *   are waiting for.
 *
 * Each model request gets the reasoning effort configured for the kind of step it follows:
 * `instruction` after an operator message, the reflection's depth after a reflection, `recover`
 * after a failure that got no reflection, `act` otherwise. The effort is written to the request
 * header of the Session log. The DeepSeek API caches prompt prefixes per effort: a request whose
 * effort differs from the previous request's reads the cache left by the last request at its own
 * effort and prefills everything after it, so a deployment that values time to first token
 * configures one effort for all kinds and lets the reflection text ask for depth.
 *
 * - `agent/inbox/claimed`: an operator message (source `user`) joins the messages quoted by later
 *   reflections of that agent and resets its reflection history and budget.
 * - `tools/post-execute`: a failure of a configured robot tool, or the failure of a program tool
 *   (`run_code`) none of whose nested failures was reflected on, attaches the reflection as an
 *   `additionalContexts` user message of source `ego-reflection`; the end of a top-level call
 *   may attach a checkpoint the same way. The outcome is otherwise the downstream decision.
 *   Inside a program the message reaches the model with the program's result.
 * - `agent/request`: sets the reasoning effort of the request after the downstream config.
 * - tool `reflect`: the model reports a difficulty no tool reported and receives a reflection as
 *   the tool result; it counts towards the budget and history but is never refused.
 * @module @steward/dsh-ego-reflection
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { createUserMessage, ReasoningEffortId, type ContentBlock, type LlmCallConfig, type UserMessage } from '@deepseek-ai/dsh-llm'
import { defineTool, type PostToolDecision, type ToolDefinition, type ToolExecution, type ToolExecutionResult } from '@deepseek-ai/dsh-tools'
import type { Candidate, EgoService, FailureReport, JsonObject, Reflection } from '@steward/dsh-ego'

declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    /** Reflection on a robot step attached by `@steward/dsh-ego-reflection`. */
    'ego-reflection': { kind: 'ego-reflection' }
  }
}

export const name = 'steward-ego-reflection'
export const inject = ['tools', 'ego']

/** How deep one reflection goes. */
export type Depth = 'shallow' | 'deep' | 'checkpoint'

/** The kind of step a model request follows, which selects its reasoning effort. */
export type StepKind = 'act' | 'instruction' | 'recover' | Depth

/** Reasoning effort per kind of step; an empty string leaves the request's effort as configured. */
export type Efforts = Record<StepKind, string>

/** Validated plugin configuration. */
export interface Config {
  /** Tools whose failures are reflected on. */
  tools: string[]
  /** Program tools (`run_code`) whose own failure is reflected on when no nested failure of the same program was. */
  programTools: string[]
  /** Reflections of any depth per operator message; later failures pass without one. */
  maxPerInstruction: number
  /** Deep reflections per operator message; within the budget a deep trigger falls back to shallow. */
  maxDeepPerInstruction: number
  /** Shallow reflections since the last deep one after which the next new failure gets a deep reflection. */
  deepAfter: number
  /** Top-level calls without a reflection after which a checkpoint review is attached; 0: never. */
  checkpointEvery: number
  /** Tools that make the robot act; a top-level call that called none of them is idle. */
  actionTools: string[]
  /** Idle top-level calls in a row after which a checkpoint review is attached; 0: never. */
  idleEvery: number
  /** Reasoning effort of the request following each kind of step; differing efforts split the provider's prefix cache. */
  effort: Efforts
  /** Objects listed per reflection (the call's targets first, then candidates). */
  candidateLimit: number
  /** Object groups listed in the scene digest of deep reflections and checkpoints. */
  sceneGroups: number
  /** Text after the last occurrence of this marker in an operator message is the instruction; empty: the whole message. */
  instructionMarker: string
  /** Longest instruction quoted, in characters; a longer one is cut at its end. */
  instructionMaxChars: number
  /** Operator messages quoted, the latest last. */
  instructionHistory: number
  /** Longest failure text quoted, in characters. */
  failureMaxChars: number
  /** Earlier failures under one operator message quoted by a reflection. */
  historyLimit: number
  /** Longest quoted earlier failure, in characters. */
  historyMaxChars: number
  /** Prediction receipts remembered to name the target of a failed `skill_run`. */
  receiptMemory: number
}

const effort = z.string().default('')

export const Config: z<Config> = z.object({
  tools: z.array(z.string()).default(['ego_query', 'ego_predict', 'skill_run', 'skill_status', 'ego_inspect',
    'ego_match_face', 'ego_annotate', 'ego_explore']),
  programTools: z.array(z.string()).default(['run_code']),
  maxPerInstruction: z.natural().min(1).default(12),
  maxDeepPerInstruction: z.natural().default(4),
  deepAfter: z.natural().min(1).default(2),
  checkpointEvery: z.natural().default(8),
  actionTools: z.array(z.string()).default(['skill_run', 'ego_explore']),
  idleEvery: z.natural().default(5),
  effort: z.object({ act: effort, instruction: effort, recover: effort, shallow: effort, deep: effort, checkpoint: effort }),
  candidateLimit: z.natural().min(1).default(6),
  sceneGroups: z.natural().min(1).default(12),
  instructionMarker: z.string().default(''),
  instructionMaxChars: z.natural().min(40).default(800),
  instructionHistory: z.natural().min(1).default(3),
  failureMaxChars: z.natural().min(40).default(400),
  historyLimit: z.natural().default(6),
  historyMaxChars: z.natural().min(40).default(160),
  receiptMemory: z.natural().min(1).default(64),
})

/** A failed robot step as reflection reads it. */
export interface Failure {
  tool: string
  arguments: JsonObject
  message: string
}

function record(value: unknown): JsonObject | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as JsonObject : undefined
}

function blockText(content: readonly ContentBlock[]): string {
  return content.map(block => block.type === 'text' ? block.text : '').join('\n').trim()
}

/**
 * The failure a tool result reports, if any: an error result, an infeasible prediction, a
 * failed skill, or a lookup that found nothing.
 * @param tool - tool name.
 * @param args - the call's arguments.
 * @param result - the dispatch outcome.
 * @returns the failure, or undefined for a successful step.
 */
export function failureOf(tool: string, args: unknown, result: Readonly<ToolExecutionResult>): Failure | undefined {
  const argumentsOf = record(args) ?? {}
  if (result.isError) return { tool, arguments: argumentsOf, message: blockText(result.content) || 'the call failed' }
  const value = record(result.value)
  if (value === undefined) return undefined
  if (tool === 'ego_predict' && value.feasible === false) {
    const issues = Array.isArray(value.issues) ? value.issues.join('; ') : ''
    return { tool, arguments: argumentsOf, message: `${String(value.skill ?? argumentsOf.skill)} is not feasible: ${issues}` }
  }
  if ((tool === 'skill_run' || tool === 'skill_status') && value.state === 'failed') {
    return { tool, arguments: argumentsOf, message: `${String(value.skill)} failed: ${String(value.message ?? '')}` }
  }
  if (tool === 'ego_query' && Array.isArray(value.objects) && value.objects.length === 0) {
    const what = typeof argumentsOf.category === 'string' ? argumentsOf.category : 'matching object'
    const near = typeof argumentsOf.near === 'string' ? ` near ${argumentsOf.near}` : ''
    return { tool, arguments: argumentsOf, message: `no ${what}${near} in the ego state` }
  }
  return undefined
}

/**
 * The instruction an operator message carries.
 * @param text - the message text.
 * @param marker - text after its last occurrence is the instruction; empty: the whole text.
 * @param maxChars - longest instruction kept; a longer one keeps its end.
 * @returns the instruction, or undefined for a message without text.
 */
export function instructionOf(text: string, marker: string, maxChars: number): string | undefined {
  const at = marker ? text.lastIndexOf(marker) : -1
  const instruction = (at >= 0 ? text.slice(at + marker.length) : text).trim()
  if (!instruction) return undefined
  return instruction.length > maxChars ? `…${instruction.slice(-maxChars)}` : instruction
}

function clip(text: string, maxChars: number): string {
  return text.length > maxChars ? `${text.slice(0, maxChars)}…` : text
}

function percent(p: number): string {
  return `${Math.round(p * 100)}%`
}

/**
 * One line describing an object and the evidence behind its name.
 * @param c - the object.
 * @returns the line, without a leading bullet.
 */
export function candidateLine(c: Candidate): string {
  const facts: string[] = []
  if (c.label) facts.push(`你标注为 ${c.label}`)
  if (c.perceived && Object.keys(c.perceived).length) {
    facts.push('感知类别 ' + Object.entries(c.perceived).map(([k, p]) => `${k} ${percent(p)}`).join(', '))
  }
  if (c.segmenter && Object.keys(c.segmenter).length) {
    facts.push('分割网络 ' + Object.entries(c.segmenter)
      .map(([k, s]) => `${k} ${s.score.toFixed(2)}${s.prompts.length ? `〔${s.prompts.join('/')}〕` : ''}`).join(', '))
  }
  if (c.extentM) facts.push(`尺寸 ${c.extentM.map(v => v.toFixed(2)).join('×')} m`)
  if (c.color) facts.push(`颜色 ${c.color}`)
  if (c.on) facts.push(`在 ${c.on} 上`)
  facts.push(`位置 (${c.position.slice(0, 3).map(v => v.toFixed(2)).join(', ')})`)
  let line = `${c.id} ${c.role && c.role !== c.category ? `${c.category}（role ${c.role}）` : c.category}：${facts.join('；')}`
  if (c.doubts.length) line += `。存疑：${c.doubts.join('；')}`
  if (c.score !== undefined && c.why?.length) line += `。吻合 ${c.score.toFixed(2)}：${c.why.join('；')}`
  return line
}

function num(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

/**
 * A digest of the Ego-State for a reflection: where the robot is and what it holds, then the
 * objects grouped by the name the robot uses for them (its label, else role, else category),
 * with their ids and what they rest on.
 * @param summary - the Ego-State summary (`ctx.ego.summary()`).
 * @param groups - most object groups listed; the largest groups come first.
 * @returns the digest lines.
 */
export function sceneDigest(summary: JsonObject, groups: number): string[] {
  const lines: string[] = []
  const self = record(summary.self)
  if (self) {
    const [x, y, yaw] = [num(self.x), num(self.y), num(self.yawDeg)]
    const where = x !== undefined && y !== undefined ? `在 (${x.toFixed(2)}, ${y.toFixed(2)})` : '位置未知'
    const heading = yaw !== undefined ? `，朝向 ${Math.round(yaw)}°` : ''
    lines.push(`机器人${where}${heading}，${typeof self.holding === 'string' ? `手里拿着 ${self.holding}` : '手里是空的'}。`)
  }
  const named = new Map<string, JsonObject[]>()
  for (const value of Array.isArray(summary.objects) ? summary.objects : []) {
    const o = record(value)
    if (o === undefined || typeof o.id !== 'string') continue
    const label = [o.label, o.role, o.category].find(v => typeof v === 'string' && v) as string | undefined
    const key = label ?? 'unknown'
    named.set(key, [...named.get(key) ?? [], o])
  }
  const sorted = [...named.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
  for (const [label, objects] of sorted.slice(0, groups)) {
    const ids = objects.map(o => String(o.id))
    const on = new Map<string, number>()
    for (const o of objects) {
      const support = typeof o.on === 'string' ? `在 ${o.on} 上` : '不在已知支撑面上'
      on.set(support, (on.get(support) ?? 0) + 1)
    }
    const where = [...on.entries()].map(([s, n]) => objects.length === 1 ? s : `${n} 个${s}`).join('，')
    const doubtful = objects.filter(o => (num(o.existence) ?? 1) < 0.5).length
    lines.push(`- ${label} ×${objects.length}（${ids.slice(0, 6).join(', ')}${ids.length > 6 ? ' …' : ''}）：${where}`
      + (doubtful ? `；其中 ${doubtful} 个存在概率低于 50%` : ''))
  }
  if (sorted.length > groups) lines.push(`- 另有 ${sorted.length - groups} 类物体未列出`)
  return lines
}

/** Where a reflection stands within the operator's messages. */
export interface ReflectionContext {
  /** The operator's messages, the latest last; empty when none was received. */
  instructions: string[]
  /** Earlier failures under the latest message, oldest first. */
  history: string[]
  /** 1 for the first reflection under the latest message. */
  ordinal: number
  /** How deep the reflection goes. */
  depth: Depth
  /** Why the reflection is deep or a checkpoint, in the model's language. */
  trigger?: string
  /** Top-level calls since the latest message. */
  steps: number
  /** The Ego-State digest of deep reflections and checkpoints. */
  scene?: string[]
}

/** Name of the tool the model calls to reflect on a difficulty no tool reported. */
export const REFLECT_TOOL = 'reflect'

/** Physical causes to examine; shared by shallow and deep reflections. */
export const CAUSES = [
  '   - 感知：物体是否被认错或叫错（感知按外观和形状命名，不知道操作员的叫法），是否只看到了一部分、被别的东西挡住、只看到了某一面；',
  '   - 物体的状态：位置、朝向、哪一面朝上、是否相互叠压或遮挡、数量是否齐全、是否离开了原来的地方；',
  '   - 参照：任务依据的参照物（印着目标的物体、标签、标记）是否被别的东西压住或遮挡、被翻面、被你自己挪动过，你看到的是否就是它；',
  '   - 动作：够不着、抓不稳、会碰撞、放置的空间不够（包括被你自己之前放下的东西占掉）；',
  '   - 任务理解：你对指令的理解和操作员的意思是否一致。',
]

/** Questions of a shallow reflection: the phenomenon, its likely causes, the cheapest check and the next step. */
export const SHALLOW_QUESTIONS = [
  '一、现象：预期发生什么，实际发生了什么？差异具体在哪里？',
  '二、原因：最可能的两三个原因，各用一句话说明依据。逐项想一想：',
  ...CAUSES,
  '三、核实：用什么观察能以最小代价区分这些原因？',
  '四、下一步：怎样改，程序里怎样检验结果；不要原样重复失败的做法。',
]

/**
 * Questions for a program that failed to compile or threw a language error: fix the code, do not
 * re-examine the physical scene.
 */
export const PROGRAM_QUESTIONS = [
  '一、错误：程序哪里写错了（语法、变量重复声明、未定义的名字、类型不符）？',
  '二、修正：怎样改写后重试；不要原样再提交同一段代码。',
]

/**
 * Whether a failure message is a program language error rather than a robot-tool or physical
 * failure. Syntax and reference errors get a short shallow reflection and never trigger a deep one.
 * @param message - the failure text.
 * @returns true when the message names a language error.
 */
export function isProgramError(message: string): boolean {
  return /\b(SyntaxError|ReferenceError|TypeError|RangeError|EvalError|URIError)\b/.test(message)
    || /Unexpected token|is not (defined|a function)|Cannot read propert|Identifier '.+' has already been declared|has already been declared/i
      .test(message)
}

/**
 * Questions of a deep reflection: macro (scene, task, language, assumptions), then micro
 * (physical causes, evidence, preconditions), then strategy (method, efficiency, plan).
 */
export const DEEP_QUESTIONS = [
  '【宏观】',
  '一、复述：用自己的话总结场景（哪些物体和任务有关、在哪里）、任务的最终状态、你把任务拆成的子任务和现在所在的子任务、到目前为止的失败。',
  '二、语言：指令里的每个指称（物体的名字、数量、位置、关系、"和……一样"这类标准），在上面的 Ego State 里对应哪个物体或事实？'
    + '哪些对应不上、可能有两种理解、或被感知叫成了别的名字？你的理解有没有偏离操作员的本意？',
  '三、假设：你的做法依赖哪些假设（物体的身份、数量、朝向、可达性、技能的实际效果、坐标系与单位、程序里的计算）？'
    + '哪些从未核实过，哪些已经被失败否定？',
  '【微观】',
  '四、原因：设想一个人在真实环境里亲手做这件事，会在哪里遇到困难？用你对物理世界的常识列出至少三个可能的原因，不要只想一个。逐项检查：',
  ...CAUSES,
  '五、证据：每个可能的原因，用什么观察能以最小代价证实或排除？先核实，再行动。',
  '六、前提：要顺利完成最终动作，世界先要处于什么状态（看得清、拿得到、彼此分开、齐全、朝向合适、留有空地……）？是否应当先创造这些条件？'
    + '你之前放下的东西（暂放的、倒出的、腾空的容器）有没有占掉后面步骤要用的台面、地面或机器人的路线，要不要先挪到边角？',
  '【策略】',
  '七、方法：退一步看，当前的方法从根本上可行吗？有没有完全不同的做法（换顺序、先创造前提、换观察位置、换技能组合）？比较代价和成功率。',
  '八、效率：哪些步骤在重复或只是微调参数？能否把剩下的工作写成一个程序循环完成，只在需要你判断时停下？',
  '九、计划：选出证据最支持的原因和方法，写出下一步做什么、预期看到什么、程序里怎样检验。',
  '十、得出"做不到"或想改用低于指令要求的做法之前，先核实这个结论的依据没有被你自己的动作改变过；'
    + '核实后确定只有操作员才能解决时（例如东西不在了、无法到达），调用 report_problem 说明。',
]

/** Question added to a deep reflection when an earlier reflection under the same message exists. */
export const RECURRING_QUESTIONS = [
  '回顾：之前反思时的假设，哪些被证实、哪些被否定？这几次失败有没有共同的模式？'
    + '有没有一直没考虑到的原因，或者需要退回到更早的步骤重新做？',
]

/** Questions of a checkpoint: progress, method, overlooked risks and the next step. */
export const CHECKPOINT_QUESTIONS = [
  '任务顺序与能力：遵循操作员及说明书要求的顺序、工具朝向和本体姿态。assemble_onto/place 只放置，不负责紧固；若要求先固定再装下一件，须用可用且有观测依据的能力完成并核验固定。做不到时报告该步骤受阻，不能用放置替代紧固或跳到下一件。',
  '确定性装配：需紧固的松散小件用 predict fasten_onto({part,body,hole?,heldHole?}) 再运行回执；仅放置可用 assemble_onto；seat_screw({part,body,hole}) 的 hole 是单孔 id，drive_screw({screw}) 核验螺丝头下沉，flip_assembly({body,turnOver?}) 抓起后观察随动、整体翻面并重新登记，turnOver=false 只抓起再放下；缺可选测量时用目录与抓取运动学估计继续并记录 warning，单颗紧固失败继续下一颗；动作完成不等于连接已观测确认；先完成任务要求的姿态准备再冻结登记；等技能终态并满足该步骤的固定要求后，只选下一松散件。本体保持同一姿态时继续使用原 body id；若刚放下的件成为下一本体，使用 releasedReference.entity 和抬高后的推算帧，不复用搬动前位置；registrationEvidence 的 low/推算来源不等于视觉核验。主动取景不足时可用经过尺寸与轴一致性检查的单/双视图，查看 registrationAttempts 的拒绝原因；任务要求翻侧/翻面后先重新观察与登记。冻结帧不依赖放件后重识别，抓起本体/支撑或明确位移才失效。releaseExecuted 与 passed、planarPassed、stableObserved 分开读；null 不算通过。宽件中心落在窄本体上只代表计划可执行，不保证稳定；依据说明书选择承托侧与完整接口，放后检查相对高度和上下表面，缺证据就报未知。',
  '装配核验：本体姿态是否符合当前任务步骤，登记后是否稳定且未搬动？平面对孔不能认证侧面/底部的三维插装，尺寸或高度变化、平面 yaw 和先验孔注册也不能证明姿态；'
    + '若发生意外倾倒，先报告并停止在其上对孔，只有侧面/顶面证据和支撑条件足够才单独恢复任务所需姿态。小件使用完整孔组、各自模板注册及抓前朝向，'
    + '不要用单孔或猜测半圈/四分之一圈绕过失败。每次释放后换视角重测本体及所有已装关系，账本区分已释放未核验、已核验和已失效；'
    + '先查看 orientation.assembly：已释放件形成且经新观测支持的组合外形可以解释变宽，孔位仍沿用装前登记帧；不要拿合并外框轴重新登记。'
    + 'envelope_observed 只核对组合外形，geometry_observed 只核对独立零件几何，都不能证明正立、半圈方向或紧固。无法解释的变化仍须停止并重观察。'
    + 'hole/heldHole 使用 screwHoles 的实际 id 或完整共享前缀，不直接填 group 元数据。技能结束后再取新状态，核验已放件及任务要求的固定后才抓下一个松散件。'
    + '最终要求多次观察下同时成立，不能把单次释放成功当成装对。',
  '一、进度：最终状态的各项条件，哪些已满足、哪些还没有？凭什么判断（列出检验过的结果，而不是计划）？',
  '二、方法：最近的程序是否在重复相似的工作或微调参数？剩下的工作能否写成一个程序循环完成，只在需要你判断时停下？',
  '三、风险：有没有被忽略的异常迹象（数量对不上、物体移位、名字可疑、结果和预期不符）？'
    + '已经放下的东西会不会让后面的步骤没有地方用，或挡住机器人的路线？',
  '四、下一步：做什么，怎样检验。',
]

/** Questions of a checkpoint after a run of programs in which the robot did not act. */
export const IDLE_QUESTIONS = [
  '一、判断：这些程序在为哪一个决定收集信息？缺的是信息，还是程序本身有错？',
  '二、重复：它们是否在反复试探同一件事（换参数、换打印方式）？把需要的观察和计算一次写全，让程序直接给出决定，而不是把数据打印出来再看。',
  '三、参照：你核实的参照物是否被遮挡、翻面或挪动过，你看到的是否就是它？',
  '四、下一步：写一个推进任务的程序（含动作和检验）；确实缺少只有操作员能给的信息时，说明缺什么。',
]

function instructionLines(instructions: readonly string[]): string[] {
  if (instructions.length === 0) return ['操作员的原始指令：（本轮没有收到操作员消息）']
  if (instructions.length === 1) return [`操作员的原始指令：「${instructions[0]}」`]
  return ['操作员的消息（从早到晚）：', ...instructions.map(i => `- 「${i}」`)]
}

function evidenceLines(evidence: Reflection | { error: string }, candidateLimit: number): string[] {
  if ('error' in evidence) return [`（无法取得 Ego 证据：${evidence.error}）`]
  const lines: string[] = []
  if (evidence.cause.key !== 'unknown') lines.push(`失败归类：${evidence.cause.label}。常见对策：${evidence.cause.advice}。`)
  const targets = evidence.targets.slice(0, candidateLimit)
  if (targets.length) lines.push('这一步涉及的物体：', ...targets.map(c => `- ${candidateLine(c)}`))
  if (evidence.missing.length) lines.push(`Ego State 里已没有：${evidence.missing.join(', ')}`)
  const others = evidence.candidates.slice(0, Math.max(0, candidateLimit - targets.length))
  const sought = evidence.sought.name || evidence.sought.size
    ? `（对照 ${[evidence.sought.name, evidence.sought.size?.map(v => `${v} m`).join('×')].filter(Boolean).join('，')}）` : ''
  if (others.length) lines.push(`可能就是指令所指、或名字存疑的其他物体${sought}：`, ...others.map(c => `- ${candidateLine(c)}`))
  return lines
}

/**
 * The reflection text for one failure or reported difficulty.
 * @param failure - the failed step; `tool` is {@link REFLECT_TOOL} for a difficulty the model reported.
 * @param context - the operator's messages, the earlier failures, the ordinal, the depth and its trigger.
 * @param evidence - the Ego Runtime's reflection evidence, or the reason it is unavailable.
 * @param config - limits.
 * @returns the message text.
 */
export function reflectionText(failure: Failure, context: ReflectionContext,
  evidence: Reflection | { error: string }, config: Pick<Config, 'candidateLimit' | 'failureMaxChars'>): string {
  const what = clip(failure.message, config.failureMaxChars)
  const deep = context.depth === 'deep'
  const programBug = !deep && isProgramError(failure.message)
  const title = deep ? '【深度反思】' : '【反思】'
  const lines = [failure.tool === REFLECT_TOOL ? `${title}你遇到的困难：${what}` : `${title}${failure.tool} 失败：${what}`]
  if (context.trigger) lines.push(`为什么深入反思：${context.trigger}`)
  lines.push(...instructionLines(context.instructions))
  if (context.history.length) lines.push('这条指令下之前的失败：', ...context.history.map(h => `- ${h}`))
  if (deep && context.scene?.length) lines.push(`当前场景（Ego State 摘要，这条指令下已运行 ${context.steps} 个程序）：`, ...context.scene)
  if (!programBug) lines.push(...evidenceLines(evidence, config.candidateLimit))
  if (deep) {
    lines.push(`这是这条指令下的第 ${context.ordinal} 次反思，是一次深度反思。先在思考中充分推敲，再分层逐条作答，每条一两句话：`)
    if (context.ordinal > 1) lines.push(...RECURRING_QUESTIONS)
    lines.push(...DEEP_QUESTIONS)
  } else if (programBug) {
    lines.push(`这是程序本身的错误，不是机器人动作失败。用一两句话回答后改程序重试：`, ...PROGRAM_QUESTIONS)
  } else {
    lines.push(`这是这条指令下的第 ${context.ordinal} 次反思。下一次行动之前，逐条回答，每条一句话：`, ...SHALLOW_QUESTIONS,
      '同一类失败再次出现时，会进入深度反思。')
  }
  if (!programBug) {
    lines.push('需要时用 ego_candidates（传入指令里的名字和大小）、ego_inspect、ego_observe 或靠近观察核实，用 ego_annotate 改正物体的名字和可供性。')
  }
  return lines.join('\n')
}

/**
 * The checkpoint text after a run of steps without reflection (`since`), or a run of steps in
 * which the robot did not act (`idle`).
 * @param context - the operator's messages, the step count, the length of the run and the scene digest.
 * @returns the message text.
 */
export function checkpointText(context: Pick<ReflectionContext, 'instructions' | 'steps' | 'scene'>
  & ({ since: number } | { idle: number })): string {
  const idle = 'idle' in context
  return [
    idle
      ? `【阶段回顾】这条指令下已运行 ${context.steps} 个程序，最近 ${context.idle} 个程序里机器人都没有动作（只在观察、计算或打印）。`
      : `【阶段回顾】这条指令下已运行 ${context.steps} 个程序，自上次反思以来的 ${context.since} 个程序没有工具报错。`,
    ...instructionLines(context.instructions),
    ...context.scene?.length ? ['当前场景（Ego State 摘要）：', ...context.scene] : [],
    '逐条回答，每条一句话，再继续：',
    ...idle ? IDLE_QUESTIONS : CHECKPOINT_QUESTIONS,
  ].join('\n')
}

interface AgentState {
  instructions: string[]
  reflections: number
  deep: number
  shallowSinceDeep: number
  /** Top-level calls since the latest operator message. */
  steps: number
  /** Top-level calls since the latest reflection. */
  quiet: number
  /** Top-level calls in a row that called no action tool. */
  idle: number
  /** An action tool was called since the last top-level call ended. */
  acted: boolean
  /** Failure signatures seen under the latest message, and whether a recurrence was reflected on deeply. */
  signatures: Map<string, { deep: boolean }>
  history: string[]
  /** `rootCallId`s of programs whose nested failures were already reflected on. */
  roots: Set<string>
  /** Deepest reflection attached since the last top-level call ended. */
  pending?: Depth
  /** A failure was seen since the last top-level call ended. */
  failed: boolean
  /** Kind of the step the next model request follows. */
  next?: StepKind
}

const RANK: Record<Depth, number> = { checkpoint: 1, shallow: 2, deep: 3 }

function signature(failure: Failure): string {
  return `${failure.tool}\u0000${failure.message.replace(/\d+(\.\d+)?/g, '#')}`
}

/**
 * Build the `reflect` tool: the model reports a difficulty no tool reported (a result that does
 * not match the goal, no progress) and receives a reflection, deep unless asked otherwise.
 * @param ask - produces the reflection text for a difficulty.
 * @returns the tool definition.
 */
export function reflectTool(ask: (difficulty: string, name: string | undefined, size: number[] | undefined,
  depth: 'shallow' | 'deep', exec: ToolExecution) => Promise<string>): ToolDefinition {
  return defineTool({
    name: REFLECT_TOOL,
    description: 'Reflect before trying again when the task is not progressing although no tool failed: a result does not '
      + 'match the goal, something expected is missing, or the same approach keeps falling short. Describe the difficulty; '
      + 'returns the operator\'s messages, earlier failures, the robot\'s evidence on the objects involved and, for a deep '
      + 'reflection, a digest of the scene, then the questions to answer before the next step. Print the returned text '
      + 'and answer it after the program ends.',
    parameters: {
      difficulty: { type: 'string', required: true, description: 'What is going wrong, with the observations that show it.' },
      name: { type: 'string', description: 'The object the instruction means, as a category word, when one is involved.' },
      size: { type: 'array', items: { type: 'number' }, description: 'Its approximate dimensions in metres, if known.' },
      depth: { type: 'string', enum: ['shallow', 'deep'], description: 'shallow: a quick check of one step; deep (default): '
        + 'question the understanding of the task, the assumptions and the method.' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => [{ type: 'text', text: v }] },
    async execute(args, exec) {
      return ask(args.difficulty, args.name, args.size, args.depth === 'shallow' ? 'shallow' : 'deep', exec)
    },
  })
}

export function apply(ctx: Context, config: Config) {
  const ego: EgoService = ctx.ego
  const tools = new Set(config.tools)
  const programs = new Set(config.programTools)
  const actions = new Set(config.actionTools)
  const states = new WeakMap<Agent, AgentState>()
  const fresh = (instructions: string[] = []): AgentState => ({ instructions, reflections: 0, deep: 0, shallowSinceDeep: 0,
    steps: 0, quiet: 0, idle: 0, acted: false, signatures: new Map(), history: [], roots: new Set(), failed: false })
  const detached = fresh()
  const receipts = new Map<string, JsonObject>()
  const stateOf = (agent: Agent | undefined): AgentState => {
    if (agent === undefined) return detached
    let state = states.get(agent)
    if (state === undefined) states.set(agent, state = fresh())
    return state
  }

  const scene = async (signal: AbortSignal): Promise<string[]> => {
    try {
      return sceneDigest(await ego.summary(signal), config.sceneGroups)
    } catch (error) {
      return [`（无法取得 Ego State：${error instanceof Error ? error.message : String(error)}）`]
    }
  }

  const note = (state: AgentState, depth: Depth, countTowardDeep = true) => {
    state.reflections += 1
    state.quiet = 0
    if (depth === 'deep') [state.deep, state.shallowSinceDeep] = [state.deep + 1, 0]
    if (depth === 'shallow' && countTowardDeep) state.shallowSinceDeep += 1
    if (state.pending === undefined || RANK[depth] > RANK[state.pending]) state.pending = depth
  }

  const reflect = async (state: AgentState, failure: Failure, report: FailureReport, depth: 'shallow' | 'deep',
    trigger: string | undefined, signal: AbortSignal, countTowardDeep = true): Promise<string> => {
    let evidence: Reflection | { error: string }
    try {
      evidence = await ego.reflect(report, signal)
    } catch (error) {
      evidence = { error: error instanceof Error ? error.message : String(error) }
    }
    const text = reflectionText(failure, {
      instructions: [...state.instructions], history: [...state.history], ordinal: state.reflections + 1, depth,
      ...trigger === undefined ? {} : { trigger }, steps: state.steps,
      ...depth === 'deep' ? { scene: await scene(signal) } : {},
    }, evidence, config)
    note(state, depth, countTowardDeep)
    state.history.push(`${failure.tool}：${clip(failure.message, config.historyMaxChars)}`)
    while (state.history.length > config.historyLimit) state.history.shift()
    return text
  }

  /** The depth a failure gets and why, or undefined when it gets no reflection. */
  const depthOf = (state: AgentState, failure: Failure):
    { depth: 'shallow' | 'deep', trigger?: string, countTowardDeep?: boolean } | undefined => {
    if (state.reflections >= config.maxPerInstruction) return undefined
    const programBug = programs.has(failure.tool) && isProgramError(failure.message)
    const deepLeft = state.deep < config.maxDeepPerInstruction
    const seen = state.signatures.get(signature(failure))
    if (seen !== undefined) {
      // A repeated language error stays a short fix; do not escalate to a deep scene review.
      if (programBug || seen.deep || !deepLeft) return undefined
      seen.deep = true
      return { depth: 'deep', trigger: '同样的失败又出现了一次：换个角度重新理解任务和场景，而不是再调整参数。' }
    }
    state.signatures.set(signature(failure), { deep: false })
    if (programBug) return { depth: 'shallow', countTowardDeep: false }
    if (deepLeft && state.shallowSinceDeep >= config.deepAfter) {
      return { depth: 'deep', trigger: `前 ${state.shallowSinceDeep} 次反思之后任务仍然受阻：检查对任务的理解、假设和方法本身。` }
    }
    return { depth: 'shallow' }
  }

  ctx.on('agent/inbox/claimed', ({ agent, message }) => {
    if (message.source.kind !== 'user') return
    const instruction = instructionOf(blockText(message.content), config.instructionMarker, config.instructionMaxChars)
    if (instruction === undefined) return
    const state = stateOf(agent)
    const instructions = [...state.instructions, instruction].slice(-config.instructionHistory)
    Object.assign(state, fresh(instructions), { next: 'instruction' })
  })

  ctx.on('agent/request', async ({ agent }, next): Promise<LlmCallConfig> => {
    const proposed = await next()
    const kind = stateOf(agent).next
    const effort = kind === undefined ? '' : config.effort[kind]
    return effort ? { ...proposed, reasoningEffort: ReasoningEffortId(effort) } : proposed
  })

  ctx.effect(() => ctx.tools.register(reflectTool(async (difficulty, sought, size, depth, exec) => {
    const state = stateOf(exec.agent)
    const failure: Failure = { tool: REFLECT_TOOL, arguments: {}, message: difficulty }
    const deep = depth === 'deep' && state.deep < config.maxDeepPerInstruction
    return reflect(state, failure, { tool: REFLECT_TOOL, message: difficulty, name: sought, size }, deep ? 'deep' : 'shallow',
      deep ? '你报告了一个没有工具报错的困难。' : undefined, exec.signal)
  })))

  ctx.on('tools/post-execute', async (exec: ToolExecution, result, next): Promise<PostToolDecision> => {
    const decision = await next()
    const program = programs.has(exec.name)
    if (!tools.has(exec.name) && !program && !actions.has(exec.name) && exec.name !== REFLECT_TOOL) return decision
    const top = exec.parent === undefined
    if (program && !top) return decision
    const value = result.isError ? undefined : record(result.value)
    if (exec.name === 'ego_predict' && typeof value?.receipt === 'string') {
      receipts.set(value.receipt, record(exec.arguments) ?? {})
      while (receipts.size > config.receiptMemory) receipts.delete(receipts.keys().next().value!)
    }
    const state = stateOf(exec.agent)
    const failure = program
      ? (result.isError && !state.roots.has(exec.rootCallId) ? failureOf(exec.name, {}, result) : undefined)
      : tools.has(exec.name) ? failureOf(exec.name, exec.arguments, result) : undefined
    if (failure !== undefined) state.failed = true
    const contexts: UserMessage[] = []
    const chosen = failure === undefined || exec.signal.aborted ? undefined : depthOf(state, failure)
    if (failure !== undefined && chosen !== undefined) {
      if (!program) state.roots.add(exec.rootCallId)
      const receipt = typeof failure.arguments.receipt === 'string' ? receipts.get(failure.arguments.receipt) : undefined
      contexts.push(createUserMessage({ content: [{ type: 'text', text: await reflect(state, failure, {
        tool: failure.tool,
        arguments: receipt ? { ...failure.arguments, params: receipt } : failure.arguments,
        message: failure.message,
      }, chosen.depth, chosen.trigger, exec.signal, chosen.countTowardDeep !== false) }],
        source: { kind: 'ego-reflection' } }))
    }
    if (actions.has(exec.name)) state.acted = true
    if (top) {
      state.steps += 1
      state.quiet += 1
      state.idle = state.acted ? 0 : state.idle + 1
      state.acted = false
      const open = state.pending === undefined && state.reflections < config.maxPerInstruction && !exec.signal.aborted
      const idle = open && config.idleEvery > 0 && state.idle >= config.idleEvery
      if (idle || (open && !state.failed && config.checkpointEvery > 0 && state.quiet >= config.checkpointEvery)) {
        const run = idle ? { idle: state.idle } : { since: state.quiet }
        const text = checkpointText({ instructions: [...state.instructions], steps: state.steps, ...run, scene: await scene(exec.signal) })
        note(state, 'checkpoint')
        state.idle = 0
        contexts.push(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'ego-reflection' } }))
      }
      state.next = state.pending ?? (state.failed ? 'recover' : 'act')
      state.pending = undefined
      state.failed = false
    }
    return contexts.length ? { ...decision, additionalContexts: [...decision.additionalContexts ?? [], ...contexts] } : decision
  })
}
