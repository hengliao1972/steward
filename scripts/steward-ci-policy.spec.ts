import { readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { runInNewContext } from 'node:vm'
import { load } from 'js-yaml'
import { describe, expect, it } from 'vitest'

const directory = resolve(import.meta.dirname, '../.github/workflows')
const upstream = 'deepseek-harness/deepseek-harness'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function workflow(file: string) {
  const value = load(readFileSync(resolve(directory, file), 'utf8'))
  if (!isRecord(value) || !isRecord(value.on) || !isRecord(value.jobs)) {
    throw new TypeError(`${file} must define events and jobs`)
  }
  return { events: value.on, jobs: value.jobs }
}

function admitted(condition: unknown, repository: string, event_name: string, cancelled = false, ref = 'refs/heads/master') {
  if (condition === undefined) return true
  if (typeof condition === 'boolean') return condition
  if (typeof condition !== 'string') throw new TypeError('Job condition must be an expression')
  const expression = condition.trim().replace(/^\$\{\{\s*|\s*\}\}$/g, '')
  return Boolean(runInNewContext(expression, {
    github: { repository, event_name, ref, event: { pull_request: { draft: false,
      head: { repo: { fork: false } }, user: { login: 'maintainer' } } } },
    inputs: { suite: 'wine' },
    vars: { MERGE_POLICY_HEAD_DIFF: 'block', MERGE_POLICY_REVIEW: 'block',
      MERGE_POLICY_MASTER_DIFF: 'block', MERGE_POLICY_PRE_MERGE: 'block' },
    needs: { master: { result: 'success' } },
    cancelled: () => cancelled,
    always: () => true,
    success: () => !cancelled,
    failure: () => false,
  }, { timeout: 1000 }))
}

describe('Steward automatic CI policy', () => {
  it('rejects every inherited automatic job in Steward forks, including push and completion events', () => {
    let checked = 0
    for (const file of readdirSync(directory).filter(file => file.endsWith('.yml') && file !== 'ci-steward.yml')) {
      const { events, jobs } = workflow(file)
      const automatic = Object.keys(events).filter(event => !['workflow_dispatch', 'workflow_call'].includes(event))
      for (const event of automatic) {
        for (const [name, job] of Object.entries(jobs)) {
          if (!isRecord(job)) throw new TypeError(`${file}/${name} must define a job`)
          for (const repository of ['Mrhonor/steward', 'hengliao1972/steward']) {
            for (const cancelled of [false, true]) {
              expect(admitted(job.if, repository, event, cancelled), `${file}/${name}: ${event} in ${repository}`)
                .toBe(false)
              checked++
            }
          }
        }
      }
    }
    expect(checked).toBeGreaterThan(100)
  })

  it('keeps upstream master validation and trusted E2E admission', () => {
    for (const name of ['python-runtime', 'windows', 'serial-linux-selfhosted', 'serial-windows']) {
      const job = workflow('ci-master.yml').jobs[name]
      if (!isRecord(job)) throw new TypeError(`Missing ${name}`)
      expect(admitted(job.if, upstream, 'push')).toBe(true)
      expect(admitted(job.if, upstream, 'push', false, 'refs/heads/topic')).toBe(false)
    }
    const e2e = workflow('e2e.yml').jobs.e2e
    if (!isRecord(e2e)) throw new TypeError('Missing E2E job')
    expect(admitted(e2e.if, upstream, 'push')).toBe(true)
    expect(admitted(e2e.if, upstream, 'schedule')).toBe(true)
  })

  it('retains upstream release, native, Sandbox and Policy jobs', () => {
    for (const [file, names, event] of [
      ['release.yml', ['dependencies', 'pack'], 'push'],
      ['release-vendor.yml', ['pack'], 'push'],
      ['node-addon-system.yml', ['matrix', 'native'], 'push'],
      ['sandbox.yml', ['sandbox-e2e', 'unit-darwin'], 'push'],
      ['expected-filenames.yml', ['expected-filenames'], 'pull_request'],
      ['pr-check-on-head-branch.yml', ['master', 'pre-merge'], 'pull_request_target'],
      ['pr-check-after-approve.yml', ['approve', 'review'], 'pull_request_target'],
      ['pr-check-on-base-branch.yml', ['recheck'], 'push'],
    ] as const) {
      for (const name of names) {
        const job = workflow(file).jobs[name]
        if (!isRecord(job)) throw new TypeError(`Missing ${file}/${name}`)
        expect(admitted(job.if, upstream, event), `${file}/${name}`).toBe(true)
      }
    }
  })

  it('runs all Steward tests with coverage on master PRs, master pushes and manual dispatches', () => {
    const { events, jobs } = workflow('ci-steward.yml')
    expect(events.push).toEqual({ branches: ['master'] })
    expect(events.pull_request).toMatchObject({ branches: ['master'], types: expect.arrayContaining(['opened', 'synchronize', 'reopened']) })
    expect(events).toHaveProperty('workflow_dispatch')
    expect(Object.keys(jobs)).toEqual(['steward-tests'])
    const job = jobs['steward-tests']
    if (!isRecord(job) || !Array.isArray(job.steps)) throw new TypeError('Missing Steward test steps')
    for (const event of ['push', 'pull_request', 'workflow_dispatch']) {
      expect(admitted(job.if, 'Mrhonor/steward', event)).toBe(true)
    }
    expect(job.defaults).toEqual({ run: { shell: 'bash', 'working-directory': 'steward' } })
    const test = job.steps.find(step => isRecord(step) && step.name === 'Run all Steward tests with coverage')
    if (!isRecord(test)) throw new TypeError('Missing pytest coverage step')
    expect(test.run).toContain('python -m pytest tests')
    expect(test.run).toContain('--cov=steward --cov-branch')
    expect(test['continue-on-error']).toBeUndefined()
  })
})
