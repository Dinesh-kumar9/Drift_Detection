import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  CheckCircle, XCircle, Clock, RefreshCw, Rocket, ShieldCheck, Ban, AlertTriangle, ScrollText, Play,
} from 'lucide-react'
import clsx from 'clsx'
import { retrainApi, type RetrainJob, type RetrainStatus } from '../api/retrain'
import { modelsApi } from '../api/models'
import { errorMessage } from '../api/client'
import { useAuth } from '../auth/AuthContext'

const STAGE_LABEL: Record<string, string> = {
  S1_ingestion: 'S1 · Ingestion', S2_preprocessing: 'S2 · Preprocessing', S3_output: 'S3 · Output',
}
const pct = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`)
const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '—')
const shortTime = (iso: string) => {
  const d = new Date(iso)
  const today = d.toDateString() === new Date().toDateString()
  return today ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

// ─── Status badge (icon + label) ─────────────────────────────────────────────
function StatusBadge({ job }: { job: RetrainJob }) {
  const s: RetrainStatus | 'promoted' | 'awaiting' =
    job.promoted_at ? 'promoted' : job.status === 'completed' ? 'awaiting' : job.status
  const map = {
    pending: { cls: 'badge-warning', icon: <Clock size={10} />, text: 'awaiting approval' },
    approved: { cls: 'bg-brand-500/20 text-brand-300 ring-1 ring-brand-500/30', icon: <RefreshCw size={10} className="animate-spin" />, text: 'queued' },
    running: { cls: 'bg-brand-500/20 text-brand-300 ring-1 ring-brand-500/30', icon: <RefreshCw size={10} className="animate-spin" />, text: 'retraining' },
    awaiting: { cls: 'badge-warning', icon: <Rocket size={10} />, text: 'candidate ready' },
    completed: { cls: 'badge-safe', icon: <CheckCircle size={10} />, text: 'completed' },
    promoted: { cls: 'badge-safe', icon: <CheckCircle size={10} />, text: 'promoted' },
    rejected: { cls: 'badge-archived', icon: <Ban size={10} />, text: 'rejected' },
    failed: { cls: 'badge-critical', icon: <XCircle size={10} />, text: 'failed' },
  }[s]
  return <span className={clsx('badge', map.cls)}>{map.icon}{map.text}</span>
}

// ─── Gate decision ───────────────────────────────────────────────────────────
function GateChecks({ job }: { job: RetrainJob }) {
  const d = job.decision
  if (!d) return null
  return (
    <div className="space-y-1.5">
      {d.checks.map((c) => (
        <div key={c.check} className="flex items-start gap-2 text-xs">
          {c.passed
            ? <CheckCircle size={13} className="text-drift-safe flex-shrink-0 mt-0.5" aria-label="passed" />
            : <XCircle size={13} className="text-drift-critical flex-shrink-0 mt-0.5" aria-label="failed" />}
          <span className="text-slate-300 font-medium w-36 flex-shrink-0">{c.check}</span>
          <span className="text-slate-500">{c.detail}</span>
        </div>
      ))}
    </div>
  )
}

// ─── Candidate vs production ────────────────────────────────────────────────
function Evaluation({ job }: { job: RetrainJob }) {
  const ev = job.evaluation
  if (!ev) return null
  const rows = [
    { label: `Production ${job.version_tag ?? ''}`, value: ev.production_error },
    { label: `Candidate ${ev.candidate_version_tag ?? ''}`, value: ev.candidate_error },
  ]
  const max = Math.max(...rows.map((r) => r.value ?? 0), 0.0001)
  const better = (ev.improvement ?? 0) > 0
  return (
    <div className="rounded-lg bg-surface border border-surface-border p-3 space-y-2">
      <p className="text-[11px] text-slate-400">
        {ev.metric} on the {ev.holdout_rows} newest labelled predictions (held out from retraining) — lower is better
      </p>
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-3 text-xs">
          <span className="w-32 text-slate-300 font-mono">{r.label}</span>
          <div className="flex-1 h-2 bg-surface-card rounded overflow-hidden">
            <div className="h-2 rounded bg-slate-400" style={{ width: `${((r.value ?? 0) / max) * 100}%` }} />
          </div>
          <span className="w-16 text-right font-mono text-slate-200">
            {ev.metric === 'error rate' ? pct(r.value) : r.value?.toFixed(3)}
          </span>
        </div>
      ))}
      <p className={clsx('text-xs font-medium', better ? 'text-drift-safe' : 'text-drift-critical')}>
        {better ? <CheckCircle size={12} className="inline mr-1" /> : <AlertTriangle size={12} className="inline mr-1" />}
        Candidate is {better ? 'better' : 'not better'} by {ev.metric === 'error rate' ? pct(Math.abs(ev.improvement ?? 0)) : Math.abs(ev.improvement ?? 0).toFixed(3)}
        {' '}· trained on {ev.training_rows.toLocaleString()} rows ({ev.training_mode === 'post_onset_only' ? 'post-drift data only' : 'original + recent data'})
      </p>
    </div>
  )
}

// ─── Job card ────────────────────────────────────────────────────────────────
function JobCard({ job, isAdmin, onChanged }: { job: RetrainJob; isAdmin: boolean; onChanged: () => void }) {
  const [rejecting, setRejecting] = useState(false)
  const [reason, setReason] = useState('')
  const act = useMutation({
    mutationFn: async (kind: 'approve' | 'promote' | 'reject') =>
      kind === 'approve' ? retrainApi.approve(job.id) : kind === 'promote' ? retrainApi.promote(job.id) : retrainApi.reject(job.id, reason),
    onSuccess: () => { setRejecting(false); onChanged() },
  })
  const awaitingPromotion = job.status === 'completed' && !job.promoted_at
  const d = job.decision

  return (
    <div className={clsx('card space-y-3', (job.status === 'pending' || awaitingPromotion) && 'border-drift-warning/40')}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold font-mono text-slate-100">
            {job.registered_model_name} <span className="text-slate-500">{job.version_tag}</span>
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">
            {job.trigger_type === 'auto' ? 'Opened automatically by a drift check' : 'Requested manually'}
            {job.trigger_reason?.root_cause && <> · root cause <span className="text-slate-300">{STAGE_LABEL[job.trigger_reason.root_cause]}</span></>}
            {job.trigger_reason?.forced && ' · gate overridden'} · {when(job.created_at)}
          </p>
          {job.trigger_reason?.note && <p className="text-xs text-slate-400 mt-1 italic">“{job.trigger_reason.note}”</p>}
        </div>
        <StatusBadge job={job} />
      </div>

      {d && (
        <div className="grid grid-cols-4 gap-2 text-center">
          {[
            ['Error at training', pct(d.baseline_error)],
            ['Error now', pct(d.current_error)],
            ['Compute cost', d.cost_estimate !== null ? `$${d.cost_estimate.toFixed(3)}` : '—'],
            ['Expected value', d.expected_value !== null ? `$${d.expected_value.toFixed(0)}` : '—'],
          ].map(([label, value]) => (
            <div key={label} className="rounded-lg bg-surface border border-surface-border py-2">
              <p className="text-[10px] text-slate-500 uppercase tracking-widest">{label}</p>
              <p className="text-sm font-mono font-bold text-slate-100 mt-0.5">{value}</p>
            </div>
          ))}
        </div>
      )}

      <GateChecks job={job} />
      {d && <p className="text-xs text-slate-300"><ShieldCheck size={12} className="inline mr-1 text-slate-500" />{d.reason}</p>}
      {job.rejection_reason && <p className="text-xs text-slate-500">{job.rejection_reason}</p>}
      {job.error && <p className="text-xs text-drift-critical font-mono break-words">{job.error}</p>}
      <Evaluation job={job} />

      {(job.status === 'pending' || awaitingPromotion) && (
        isAdmin ? (
          <div className="space-y-2">
            {rejecting ? (
              <div className="flex gap-2">
                <input className="input text-xs flex-1" placeholder="Reason (required)" value={reason} onChange={(e) => setReason(e.target.value)} />
                <button className="btn-danger text-xs" disabled={reason.trim().length < 3 || act.isPending} onClick={() => act.mutate('reject')}>Confirm</button>
                <button className="btn-secondary text-xs" onClick={() => setRejecting(false)}>Cancel</button>
              </div>
            ) : (
              <div className="flex gap-2">
                {job.status === 'pending' ? (
                  <button className="btn-primary flex-1 justify-center text-sm" disabled={act.isPending} onClick={() => act.mutate('approve')}>
                    <Play size={14} />Approve retraining
                  </button>
                ) : (
                  <button className="btn-primary flex-1 justify-center text-sm" disabled={act.isPending} onClick={() => act.mutate('promote')}>
                    <Rocket size={14} />Promote candidate to production
                  </button>
                )}
                <button className="btn-secondary text-sm" onClick={() => setRejecting(true)}>
                  <Ban size={14} />{job.status === 'pending' ? 'Reject' : 'Decline'}
                </button>
              </div>
            )}
            {act.isError && <p className="text-xs text-drift-critical">{errorMessage(act.error)}</p>}
          </div>
        ) : (
          <p className="text-xs text-slate-500"><ShieldCheck size={12} className="inline mr-1" />An admin must {job.status === 'pending' ? 'approve this retrain' : 'promote or decline this candidate'}.</p>
        )
      )}
    </div>
  )
}

// ─── Manual trigger ──────────────────────────────────────────────────────────
function TriggerForm({ onChanged }: { onChanged: () => void }) {
  const [modelId, setModelId] = useState('')
  const [note, setNote] = useState('')
  const [force, setForce] = useState(false)
  const { data: production } = useQuery({ queryKey: ['models', 'production'], queryFn: () => modelsApi.list({ status: 'production' }) })
  const trigger = useMutation({ mutationFn: () => retrainApi.trigger(modelId, note, force), onSuccess: onChanged })
  const id = modelId || production?.items[0]?.id || ''
  return (
    <div className="card space-y-3">
      <p className="text-sm font-medium text-slate-300">Request a retrain manually</p>
      <select aria-label="Model" className="input text-xs" value={id} onChange={(e) => setModelId(e.target.value)}>
        {(production?.items ?? []).map((m) => (
          <option key={m.id} value={m.id}>{m.registered_model_name} · {m.model_type} {m.version_tag}</option>
        ))}
      </select>
      <input className="input text-xs" placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
      <label className="flex items-center gap-2 text-xs text-slate-400">
        <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
        Override the cost-benefit gate (still needs admin approval)
      </label>
      <button className="btn-secondary w-full justify-center text-sm" disabled={!id || trigger.isPending}
        onClick={() => { setModelId(id); trigger.mutate() }}>
        <RefreshCw size={14} />Evaluate & request
      </button>
      {trigger.isSuccess && (
        <p className="text-xs text-slate-400">
          {trigger.data.status === 'pending' ? 'Request opened — awaiting admin approval.' : `Not opened: ${trigger.data.rejection_reason}`}
        </p>
      )}
      {trigger.isError && <p className="text-xs text-drift-critical">{errorMessage(trigger.error)}</p>}
    </div>
  )
}

// ─── Audit trail ─────────────────────────────────────────────────────────────
function AuditTrail() {
  const { data } = useQuery({ queryKey: ['audit'], queryFn: () => retrainApi.audit(40), refetchInterval: 5000 })
  return (
    <div className="card">
      <p className="text-sm font-medium text-slate-300 mb-3 inline-flex items-center gap-2"><ScrollText size={14} />Audit trail</p>
      {!data?.length ? (
        <p className="text-center py-6 text-sm text-slate-600">No events yet</p>
      ) : (
        <table className="w-full text-xs table-fixed">
          <colgroup><col className="w-[4.5rem]" /><col className="w-[34%]" /><col className="w-[36%]" /><col /></colgroup>
          <thead><tr className="text-slate-500 text-left"><th className="py-1.5">When</th><th>Who</th><th>Action</th><th>Detail</th></tr></thead>
          <tbody>
            {data.map((log) => {
              const m = log.metadata ?? {}
              const detail = (m.gate ?? m.reason ?? m.version_tag ?? m.error ?? m.registered_model ?? m.fault ?? '') as string
              return (
                <tr key={log.id} className="border-t border-surface-border/50 text-slate-300">
                  <td className="py-1.5 font-mono text-slate-500 pr-2" title={new Date(log.created_at).toLocaleString()}>{shortTime(log.created_at)}</td>
                  <td className="pr-2 truncate" title={log.actor_email ?? ''}>{log.actor_email ?? '—'}</td>
                  <td className="font-mono pr-2 truncate" title={log.action}>{log.action}</td>
                  <td className="text-slate-500 truncate" title={String(detail)}>{String(detail)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

// ─── Page ────────────────────────────────────────────────────────────────────
export default function RetrainApproval() {
  const queryClient = useQueryClient()
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'
  const canRequest = user?.role === 'admin' || user?.role === 'engineer'

  const { data: jobs, isLoading } = useQuery({
    queryKey: ['retrain-jobs'],
    queryFn: retrainApi.jobs,
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === 'approved' || j.status === 'running') ? 3000 : 10000),
  })
  const refresh = () => {
    for (const key of ['retrain-jobs', 'audit', 'models', 'registered-models', 'obs-overview']) {
      queryClient.invalidateQueries({ queryKey: [key] })
    }
  }

  const all = jobs ?? []
  const decide = all.filter((j) => j.status === 'pending' || (j.status === 'completed' && !j.promoted_at))
  const running = all.filter((j) => j.status === 'approved' || j.status === 'running')
  const history = all.filter((j) => !decide.includes(j) && !running.includes(j))

  return (
    <div className="animate-slide-up max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="section-title">Retrain Approval Console</h1>
          <p className="text-sm text-slate-500 mt-1">
            Drift opens a retrain request only when the cost-benefit gate agrees — then an admin approves the spend and promotes the result
          </p>
        </div>
        <div className="flex gap-3">
          {[['Needs decision', decide.length], ['Retraining', running.length], ['History', history.length]].map(([label, value]) => (
            <div key={label} className="card py-3 px-5 text-center min-w-[90px]">
              <p className="metric-value text-2xl">{value}</p>
              <p className="metric-label mt-0.5">{label}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-5 gap-6">
        <div className="col-span-3 space-y-4">
          {isLoading ? (
            <div className="card text-center py-12 text-slate-600 text-sm">Loading…</div>
          ) : decide.length + running.length === 0 ? (
            <div className="card text-center py-12">
              <CheckCircle size={28} className="text-slate-700 mx-auto mb-3" />
              <p className="text-slate-500 text-sm">Nothing waiting for a decision</p>
              <p className="text-slate-600 text-xs mt-1">Inject concept drift on the Observability page to see a retrain request appear</p>
            </div>
          ) : (
            [...running, ...decide].map((j) => <JobCard key={j.id} job={j} isAdmin={isAdmin} onChanged={refresh} />)
          )}

          {history.length > 0 && (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">History</p>
              {history.map((j) => <JobCard key={j.id} job={j} isAdmin={isAdmin} onChanged={refresh} />)}
            </div>
          )}
        </div>

        <div className="col-span-2 space-y-6">
          {canRequest && <TriggerForm onChanged={refresh} />}
          <AuditTrail />
        </div>
      </div>
    </div>
  )
}
