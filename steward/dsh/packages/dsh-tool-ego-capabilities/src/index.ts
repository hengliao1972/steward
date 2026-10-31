/**
 * The robot's own capability library.
 *
 * The Ego Harness writes programs for `run_code` (PTC mode) out of the robot tools. A program
 * that solved a task can be saved under a name with a description and parameters; later tasks
 * list the library, fetch a capability's code and run it with new arguments. The library lives
 * in one directory of the robot's Harness home, one JSON file per capability, so it outlives
 * sessions and scenarios. Code is stored as written and only ever runs inside `run_code`.
 *
 * Capabilities are layered: a `skill` is one reusable operation on any objects given in its
 * arguments, a `recipe` composes skills for one kind of task. Saving rejects code that names
 * Ego-State objects or holds many decimal constants, which belong in the arguments. Every
 * `run_code` program that fetched a capability counts as one run of it, succeeded when the
 * program ended without an error: a capability becomes `verified` once it succeeded in
 * `verifyAfterRuns` different Harness runs and `deprecated` after `deprecateAfterFailures`
 * failures in a row. A run is one turn of the Harness session; an id is read from a
 * `run_code` call's arguments, else each turn is counted as its own run. Saving different code
 * starts a new version as a `candidate`. Deleting moves the record to `forgotten/` with the
 * reason, out of the list.
 * @module @steward/dsh-tool-ego-capabilities
 */

import { randomUUID } from 'node:crypto'
import { mkdir, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { defineTool, type InferValue, type ToolDefinition } from '@deepseek-ai/dsh-tools'

export const name = 'steward-tool-ego-capabilities'
export const inject = ['tools']

/** Validated plugin configuration. */
export interface Config {
  /** Directory holding the capability files (created on first save). */
  root: string
  /** Different Harness runs a capability must succeed in to become `verified`. */
  verifyAfterRuns: number
  /** Failures in a row that make a capability `deprecated`. */
  deprecateAfterFailures: number
  /** Distinct decimal constants a capability's code may hold before saving is refused. */
  maxDecimalLiterals: number
}

export const Config: z<Config> = z.object({
  root: z.string().required(),
  verifyAfterRuns: z.natural().min(1).default(2),
  deprecateAfterFailures: z.natural().min(1).default(3),
  maxDecimalLiterals: z.natural().default(6),
})

/** `skill`: one reusable operation on the objects in its arguments; `recipe`: skills composed for one kind of task. */
export type CapabilityLayer = 'skill' | 'recipe'
/** `candidate`: saved, not yet proven; `verified`: succeeded in several runs; `deprecated`: failing, to be fixed or deleted. */
export type CapabilityStatus = 'candidate' | 'verified' | 'deprecated'

/** Outcomes of the current version's runs. */
export interface CapabilityRuns {
  succeeded: number
  failed: number
  /** Failures since the last success. */
  failedInARow: number
  /** Ids of the Harness runs in which it succeeded. */
  succeededIn: string[]
  /** The latest run: when, and its error if it failed. */
  last?: { at: string; ok: boolean; error?: string }
}

/** One saved capability. */
export interface Capability {
  name: string
  layer: CapabilityLayer
  description: string
  /** What must hold in the world before it runs. */
  requires: string
  /** What holds in the world after it succeeded. */
  ensures: string
  /** What each argument of `args` means. */
  parameters: Record<string, string>
  /** Body of an async function of `(tools, args)` in TypeScript erasable syntax. */
  code: string
  /** Counts the code changes; a new version starts as a candidate. */
  version: number
  status: CapabilityStatus
  runs: CapabilityRuns
  createdAt: string
  updatedAt: string
  /** Times it was fetched with `capability_get`. */
  uses: number
}

/** The fields a save supplies. */
export type CapabilityInput = Pick<Capability, 'name' | 'layer' | 'description' | 'requires' | 'ensures' | 'parameters' | 'code'>

/** The promotion and saving rules of a store. */
export type CapabilityRules = Pick<Config, 'verifyAfterRuns' | 'deprecateAfterFailures' | 'maxDecimalLiterals'>

const NAME = /^[a-z][a-z0-9_-]{0,63}$/
const ENTITY_ID = /\be:\d+\b/g
const DECIMAL = /(?<![\w.])\d+\.\d+(?![\w.])/g
const STATUS_ORDER: Record<CapabilityStatus, number> = { verified: 0, candidate: 1, deprecated: 2 }
const DEFAULT_RULES: CapabilityRules = { verifyAfterRuns: 2, deprecateAfterFailures: 3, maxDecimalLiterals: 6 }

function freshRuns(): CapabilityRuns {
  return { succeeded: 0, failed: 0, failedInARow: 0, succeededIn: [] }
}

/**
 * Why code cannot be saved as a capability: it names Ego-State objects, or holds more distinct
 * decimal constants than allowed (positions and sizes of one task belong in the arguments).
 * @param code - the capability code.
 * @param maxDecimalLiterals - distinct decimal constants allowed.
 * @returns the reasons, empty when it may be saved.
 */
export function taskSpecifics(code: string, maxDecimalLiterals: number): string[] {
  const reasons: string[] = []
  const ids = [...new Set(code.match(ENTITY_ID) ?? [])]
  if (ids.length) reasons.push(`it names objects of one task (${ids.join(', ')}); pass them in args`)
  const decimals = [...new Set(code.match(DECIMAL) ?? [])]
  if (decimals.length > maxDecimalLiterals) {
    reasons.push(`it holds ${decimals.length} decimal constants (${decimals.slice(0, 10).join(', ')}), more than `
      + `${maxDecimalLiterals}; take positions and sizes from args or from the tools`)
  }
  return reasons
}

/** Capability files in one directory; writes replace a file atomically. */
export class CapabilityStore {
  constructor(readonly root: string, readonly rules: CapabilityRules = DEFAULT_RULES) {}

  private path(capability: string): string {
    if (!NAME.test(capability)) {
      throw new Error(`capability names are 1-64 lowercase letters, digits, "_" or "-", starting with a letter: ${JSON.stringify(capability)}`)
    }
    return join(this.root, `${capability}.json`)
  }

  private async write(record: Capability): Promise<void> {
    const file = this.path(record.name)
    await mkdir(this.root, { recursive: true })
    await writeFile(`${file}.tmp`, JSON.stringify(record, null, 2))
    await rename(`${file}.tmp`, file)
  }

  /**
   * Read one capability.
   * @param capability - its name.
   * @returns the record, or undefined when there is none.
   */
  async read(capability: string): Promise<Capability | undefined> {
    try {
      return JSON.parse(await readFile(this.path(capability), 'utf8')) as Capability
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
      throw error
    }
  }

  /**
   * Create or replace a capability. Unchanged code keeps its version, status and runs; changed
   * code starts the next version as a candidate.
   * @param input - name, layer, description, conditions, parameters and code.
   * @returns the stored record.
   * @throws Error when the code holds task specifics (see {@link taskSpecifics}).
   */
  async save(input: CapabilityInput): Promise<Capability> {
    const reasons = taskSpecifics(input.code, this.rules.maxDecimalLiterals)
    if (reasons.length) throw new Error(`not saved: ${reasons.join('; ')}`)
    const now = new Date().toISOString()
    const before = await this.read(input.name)
    const same = before?.code === input.code
    const record: Capability = {
      ...input,
      version: before === undefined ? 1 : same ? before.version : before.version + 1,
      status: same && before !== undefined ? before.status : 'candidate',
      runs: same && before !== undefined ? before.runs : freshRuns(),
      createdAt: before?.createdAt ?? now,
      updatedAt: now,
      uses: before?.uses ?? 0,
    }
    await this.write(record)
    return record
  }

  /**
   * Fetch a capability for use and count the use.
   * @param capability - its name.
   * @returns the record, or undefined when there is none.
   */
  async use(capability: string): Promise<Capability | undefined> {
    const record = await this.read(capability)
    if (record === undefined) return undefined
    record.uses += 1
    await this.write(record)
    return record
  }

  /**
   * Count one run of a capability and promote or deprecate it by the store's rules.
   * @param capability - its name.
   * @param run - id of the Harness run it ran in.
   * @param ok - whether the program that ran it ended without an error.
   * @param error - the program's error message when it failed.
   * @returns the updated record, or undefined when it was deleted meanwhile.
   */
  async record(capability: string, run: string, ok: boolean, error?: string): Promise<Capability | undefined> {
    const record = await this.read(capability)
    if (record === undefined) return undefined
    const runs = record.runs
    runs.last = { at: new Date().toISOString(), ok, ...(error === undefined ? {} : { error: error.slice(0, 300) }) }
    if (ok) {
      runs.succeeded += 1
      runs.failedInARow = 0
      if (!runs.succeededIn.includes(run)) runs.succeededIn.push(run)
      if (runs.succeededIn.length >= this.rules.verifyAfterRuns) record.status = 'verified'
      else if (record.status === 'deprecated') record.status = 'candidate'
    } else {
      runs.failed += 1
      runs.failedInARow += 1
      if (runs.failedInARow >= this.rules.deprecateAfterFailures) record.status = 'deprecated'
    }
    await this.write(record)
    return record
  }

  /**
   * Forget a capability: move it out of the library into `forgotten/` with the reason.
   * @param capability - its name.
   * @param reason - why it is no longer wanted.
   * @returns the forgotten record, or undefined when there is none.
   */
  async forget(capability: string, reason: string): Promise<Capability | undefined> {
    const record = await this.read(capability)
    if (record === undefined) return undefined
    const dir = join(this.root, 'forgotten')
    await mkdir(dir, { recursive: true })
    const stamp = new Date().toISOString().replace(/[:.]/g, '-')
    await writeFile(join(dir, `${capability}-${stamp}.json`), JSON.stringify({ ...record, forgottenBecause: reason }, null, 2))
    await rm(this.path(capability))
    return record
  }

  /**
   * Every capability without its code: verified first, deprecated last, recipes before skills,
   * then most recently changed first.
   * @returns each record without `code`.
   */
  async list(): Promise<Omit<Capability, 'code'>[]> {
    let files: string[]
    try {
      files = await readdir(this.root)
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []
      throw error
    }
    const records: Omit<Capability, 'code'>[] = []
    for (const file of files.filter(f => f.endsWith('.json')).sort()) {
      const { code: _code, ...rest } = JSON.parse(await readFile(join(this.root, file), 'utf8')) as Capability
      records.push(rest)
    }
    return records.sort((a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status]
      || (a.layer === b.layer ? 0 : a.layer === 'recipe' ? -1 : 1)
      || b.updatedAt.localeCompare(a.updatedAt))
  }
}

/** The part of a tool execution {@link CapabilityRunLedger} reads. */
export interface ObservedExecution {
  readonly callId: string
  readonly rootCallId?: string
  readonly name: string
  readonly arguments: unknown
}

/** The part of a tool result {@link CapabilityRunLedger} reads. */
export interface ObservedResult {
  readonly isError: boolean
  readonly error?: { message: string }
}

/**
 * Attributes `run_code` outcomes to the capabilities each program fetched: a fetch inside a
 * program is remembered under the program's call id, and the program's final result counts as
 * one run of each capability it fetched.
 */
export class CapabilityRunLedger {
  private readonly fetched = new Map<string, Set<string>>()
  private readonly runs = new Map<string, string>()

  /**
   * @param store - the library to record runs in.
   * @param run - id prefix for runs not identified by the session (defaults to a fresh one).
   */
  constructor(readonly store: CapabilityStore, readonly run: string = randomUUID()) {}

  /**
   * Note one final tool result.
   * @param exec - the execution.
   * @param result - its final result.
   * @returns the records updated by a finished program (empty for other calls).
   */
  async observe(exec: ObservedExecution, result: ObservedResult): Promise<Capability[]> {
    if (exec.name === 'capability_get' && exec.rootCallId !== undefined && !result.isError) {
      const name = (exec.arguments as { name?: unknown } | null)?.name
      if (typeof name === 'string') {
        const names = this.fetched.get(exec.rootCallId) ?? new Set<string>()
        names.add(name)
        this.fetched.set(exec.rootCallId, names)
      }
      return []
    }
    if (exec.name !== 'run_code') return []
    const names = this.fetched.get(exec.callId)
    if (names === undefined) return []
    this.fetched.delete(exec.callId)
    const updated: Capability[] = []
    for (const name of names) {
      const record = await this.store.record(name, this.runOf(exec), !result.isError,
        result.isError ? result.error?.message : undefined)
      if (record !== undefined) updated.push(record)
    }
    return updated
  }

  /** The session's run of a `run_code` call: its turn, or an id from its arguments, else a fresh
   * id per call. One turn is one task's unit of work; several turns of one session are separate
   * runs, so a capability proven in two turns is verified, not only in two sessions. */
  private runOf(exec: ObservedExecution): string {
    let run = this.runs.get(exec.callId)
    if (run === undefined) {
      const args = (exec.arguments ?? {}) as Record<string, unknown>
      let session: string | null = null
      if (typeof args.sessionId === 'string') session = args.sessionId
      else if (typeof args.session === 'string') session = args.session
      const turn = typeof args.turn === 'number' ? `turn-${args.turn}` : null
      run = session === null ? `${this.run}/${exec.callId}` : turn === null ? session : `${session}/${turn}`
      this.runs.set(exec.callId, run)
    }
    return run
  }
}

const JSON_OUTPUT = { type: 'json' } as const
type Json = InferValue<typeof JSON_OUTPUT>

function json(value: unknown): Json {
  return JSON.parse(JSON.stringify(value)) as Json
}

function text(value: unknown): [{ type: 'text'; text: string }] {
  return [{ type: 'text', text: JSON.stringify(value, null, 1) }]
}

/**
 * Build the capability tool definitions over a store.
 * @param store - the robot's capability files.
 * @returns tool definitions in registration order.
 */
export function capabilityTools(store: CapabilityStore): ToolDefinition[] {
  return [
    defineTool({
      name: 'capability_list',
      description: 'List the capabilities you saved in earlier tasks, without their code: programs you wrote that '
        + 'combine the robot tools to do something no single tool does. Check it before writing a new program; a '
        + 'capability that fits saves you from solving the problem again. Each has a layer (recipe: skills composed '
        + 'for one kind of task; skill: one reusable operation), a status (verified: it succeeded in several runs; '
        + 'candidate: not yet proven; deprecated: it kept failing, fix it or delete it), what it requires and ensures, '
        + 'and the outcomes of its runs. Verified ones come first, recipes before skills.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute() {
        return json({ capabilities: await store.list() })
      },
    }),
    defineTool({
      name: 'capability_get',
      description: 'Fetch a saved capability, including its code. Run it inside run_code: '
        + '`const cap = await tools.capability_get({ name }); '
        + 'const run = new (Object.getPrototypeOf(async function () {}).constructor)("tools", "args", cap.code); '
        + 'return await run(tools, { ...arguments })`. The program that fetched it counts as one run of it: '
        + 'succeeded when the program ends without an error, failed when it throws.',
      parameters: { name: { type: 'string', required: true, description: 'Capability name.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args) {
        const record = await store.use(args.name)
        if (record === undefined) throw new Error(`no capability named ${JSON.stringify(args.name)}; see capability_list`)
        return json(record)
      },
    }),
    defineTool({
      name: 'capability_save',
      description: 'Save a program as a named capability of this robot so later tasks can reuse it, or replace an '
        + 'existing one with an improved version (changed code starts a new version as a candidate). Save it after it '
        + 'has worked. Prefer small skills that work on any objects of a kind (spread objects out, turn a flat object '
        + 'face up, find where a piece belongs in a picture, lay objects out on a grid), and a thin recipe that '
        + 'composes them with capability_get. Write the code as the body of an async function of (tools, args) that '
        + 'calls the same tools as your run_code programs, returns its result and throws an Error when it cannot '
        + 'achieve what it ensures. Everything specific to one task belongs in args: saving is refused when the code '
        + 'names objects (e:123) or holds many decimal constants.',
      parameters: {
        name: { type: 'string', required: true, description: 'Lowercase name, for example "lay_out_on_grid".' },
        layer: { type: 'string', enum: ['skill', 'recipe'], required: true, description: 'skill: one reusable operation; recipe: skills composed for one kind of task.' },
        description: { type: 'string', required: true, description: 'What it does and what it returns.' },
        requires: { type: 'string', required: true, description: 'What must hold in the world before it runs.' },
        ensures: { type: 'string', required: true, description: 'What holds in the world after it succeeded.' },
        parameters: { type: 'object', additionalProperties: true, description: 'What each field of args means, for example {"lid": "id of the lid showing the picture"}.' },
        code: { type: 'string', required: true, description: 'Body of an async function of (tools, args).' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args) {
        const parameters = Object.fromEntries(Object.entries((args.parameters ?? {}) as Record<string, unknown>)
          .map(([key, value]) => [key, String(value)]))
        const { code: _code, ...saved } = await store.save({
          name: args.name, layer: args.layer as CapabilityLayer, description: args.description,
          requires: args.requires, ensures: args.ensures, parameters, code: args.code,
        })
        return json(saved)
      },
    }),
    defineTool({
      name: 'capability_delete',
      description: 'Forget a capability that is wrong, superseded by a better or more general one, or no longer '
        + 'useful: it leaves the list and can no longer be fetched. Merge near-duplicates into one capability and '
        + 'delete the rest.',
      parameters: {
        name: { type: 'string', required: true, description: 'Capability name.' },
        reason: { type: 'string', required: true, description: 'Why it is forgotten, for example "superseded by lay_out_on_grid".' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args) {
        const record = await store.forget(args.name, args.reason)
        if (record === undefined) throw new Error(`no capability named ${JSON.stringify(args.name)}; see capability_list`)
        return json({ forgotten: record.name, reason: args.reason })
      },
    }),
  ]
}

export function apply(ctx: Context, config: Config): void {
  const store = new CapabilityStore(config.root, config)
  for (const tool of capabilityTools(store)) ctx.tools.register(tool)
  const ledger = new CapabilityRunLedger(store)
  ctx.on('tools/result', (exec, result) => {
    ledger.observe(exec, result).catch((error: unknown) => {
      // The tool result stands; only the library's run counts miss this run.
      console.error(`capability run not recorded: ${error instanceof Error ? error.message : String(error)}`)
    })
  })
}
