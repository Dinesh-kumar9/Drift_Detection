import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  Activity, AlertTriangle, CheckCircle, CircleDashed, FlaskConical, Play, Table2, Zap,
} from 'lucide-react'
import clsx from 'clsx'
import {
  observabilityApi, STAGES,
  type AttributionReport, type CheckOutcome, type FaultType, type OverviewItem, type Stage, type StageStatus,
} from '../api/observability'
import { modelsApi } from '../api/models'
import { errorMessage } from '../api/client'

// Validated categorical palette (dark surface #1e293b, all pairs): no status hues,
// so "drifted" red / "stable" green stay unambiguous. Identity never relies on color
// alone — every series also carries a legend entry and a direct end label.
const STAGE_META: Record<Stage, { label: string; short: string; color: string; what: string }> = {
  S1_ingestion: { label: 'S1 · Ingestion', short: 'S1', color: '#2f6fd0', what: 'raw inputs vs training data' },
  S2_preprocessing: { label: 'S2 · Preprocessing', short: 'S2', color: '#36a4c0', what: 'model features + pipeline consistency' },
  S3_output: { label: 'S3 · Output', short: 'S3', color: '#d55181', what: 'predictions, confidence, error rate' },
}

const FAULTS: { value: FaultType; label: string; hint: string }[] = [
  { value: 's1_shift', label: 'S1 · Upstream data shift', hint: 'a raw column shifts (e.g. unit change upstream)' },
  { value: 's1_nulls', label: 'S1 · Column arrives empty', hint: 'a raw column becomes null (broken join)' },
  { value: 's2_shift', label: 'S2 · Preprocessing bug', hint: 'transformed feature offset; raw inputs untouched' },
  { value: 's3_concept', label: 'S3 · Concept drift', hint: 'same inputs, ground truth flips' },
  { value: 'none', label: 'Healthy traffic', hint: 'no fault — should stay quiet' },
]

const fmtTime = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleTimeString() : '—')

// ─── Status badge (status palette — always icon + label) ─────────────────────
function HealthBadge({ status }: { status: OverviewItem['status'] | 'drift' | 'stable' }) {
  const map = {
    drifting: { cls: 'badge-critical', icon: <AlertTriangle size={10} />, text: 'drifting' },
    drift: { cls: 'badge-critical', icon: <AlertTriangle size={10} />, text: 'drift' },
    healthy: { cls: 'badge-safe', icon: <CheckCircle size={10} />, text: 'healthy' },
    stable: { cls: 'badge-safe', icon: <CheckCircle size={10} />, text: 'stable' },
    not_checked: { cls: 'badge-archived', icon: <CircleDashed size={10} />, text: 'not checked' },
    no_baseline: { cls: 'badge-warning', icon: <AlertTriangle size={10} />, text: 'no baseline' },
  }[status]
  return <span className={clsx('badge', map.cls)}>{map.icon}{map.text}</span>
}

function StageSwatch({ stage }: { stage: Stage }) {
  return <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: STAGE_META[stage].color }} />
}

// ─── Model strip (FR17: every production model on one screen) ────────────────
function ModelStrip({ items, selected, onSelect }: {
  items: OverviewItem[]; selected: string | null; onSelect: (id: string) => void
}) {
  return (
    <div className="grid grid-cols-3 gap-3">
      {items.map((m) => (
        <button
          key={m.model_version_id}
          onClick={() => onSelect(m.model_version_id)}
          className={clsx(
            'card text-left py-3 transition-colors',
            selected === m.model_version_id ? 'border-brand-500/60 bg-brand-600/10' : 'hover:border-surface-muted',
          )}
        >
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-mono font-semibold text-slate-100 truncate">{m.registered_model_name}</p>
            <HealthBadge status={m.status} />
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            {m.model_type} {m.version_tag} · SLA {m.sla_tier} · {m.predictions_24h.toLocaleString()} predictions/24h
          </p>
          <div className="flex gap-3 mt-2 text-[11px] font-mono text-slate-400">
            {STAGES.map((s) => (
              <span key={s} className="inline-flex items-center gap-1">
                <StageSwatch stage={s} />{STAGE_META[s].short} {m.stages[s]?.score.toFixed(2) ?? '—'}
                {m.stages[s]?.drifted && <AlertTriangle size={10} className="text-drift-critical" aria-label="drifted" />}
              </span>
            ))}
          </div>
        </button>
      ))}
    </div>
  )
}

// ─── Stage card ──────────────────────────────────────────────────────────────
function StageCard({ stage, status, threshold }: { stage: Stage; status?: StageStatus; threshold: number }) {
  const meta = STAGE_META[stage]
  const score = status?.score
  const diverged = (status?.details?.signals as { diverged?: boolean } | undefined)?.diverged
  const errorDrift = (status?.details?.signals as { error_drift?: boolean } | undefined)?.error_drift
  return (
    <div className={clsx('card', status?.drifted && 'border-drift-critical/40')}>
      <div className="flex items-center justify-between">
        <p className="text-xs font-mono text-slate-400 uppercase tracking-widest inline-flex items-center gap-2">
          <StageSwatch stage={stage} />{meta.label}
        </p>
        {status && <HealthBadge status={status.drifted ? 'drift' : 'stable'} />}
      </div>
      <p className="metric-value mt-3">{score !== undefined ? score.toFixed(3) : '—'}</p>
      <p className="metric-label mt-1">Drift score (threshold {threshold})</p>
      <div className="mt-3 h-1 rounded-full bg-surface-border overflow-hidden">
        <div className="h-1 rounded-full" style={{ width: `${Math.min((score ?? 0) * 100, 100)}%`, background: meta.color }} />
      </div>
      <dl className="mt-3 space-y-1 text-[11px]">
        <div className="flex justify-between"><dt className="text-slate-500">Watches</dt><dd className="text-slate-300">{meta.what}</dd></div>
        <div className="flex justify-between"><dt className="text-slate-500">Top feature</dt><dd className="text-slate-300 font-mono truncate max-w-[60%]">{status?.top_feature ?? '—'}</dd></div>
        <div className="flex justify-between"><dt className="text-slate-500">Onset</dt><dd className="text-slate-300 font-mono">{fmtTime(status?.onset)}</dd></div>
        <div className="flex justify-between"><dt className="text-slate-500">Samples</dt><dd className="text-slate-300 font-mono">{status?.n_samples ?? '—'}</dd></div>
      </dl>
      {diverged && <p className="mt-2 text-[11px] text-drift-critical">Features diverge from the training-time transform</p>}
      {errorDrift && <p className="mt-2 text-[11px] text-drift-critical">Error rate rose against ground truth</p>}
    </div>
  )
}

// ─── Trend chart ─────────────────────────────────────────────────────────────
function TrendChart({ series, threshold }: { series: { t: string; [k: string]: number | string }[]; threshold: number }) {
  const [showTable, setShowTable] = useState(false)
  const last = series.length - 1
  // Dodge end labels whose lines finish close together so they never overlap
  const labelDy = useMemo(() => {
    const finals = STAGES.map((s) => ({ s, v: Number(series[last]?.[s] ?? 0) })).sort((a, b) => b.v - a.v)
    const dy: Partial<Record<Stage, number>> = {}
    finals.forEach(({ s, v }, i) => {
      const prev = finals[i - 1]
      dy[s] = prev && prev.v - v < 0.06 ? (dy[prev.s] ?? 0) + 12 : 0
    })
    return dy
  }, [series, last])
  const endLabel = (stage: Stage) => (props: { x?: number | string; y?: number | string; index?: number }) =>
    props.index === last ? (
      <text x={Number(props.x) + 8} y={Number(props.y) + 4 + (labelDy[stage] ?? 0)} fontSize={11} fill="#cbd5e1">
        {STAGE_META[stage].short}
      </text>
    ) : <g />

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-sm font-medium text-slate-300">Drift score per stage over time</p>
          <p className="text-[11px] text-slate-500">largest effect size per check · dashed line = drift threshold ({threshold})</p>
        </div>
        <button onClick={() => setShowTable((v) => !v)} className="text-[11px] text-slate-400 hover:text-slate-200 inline-flex items-center gap-1">
          <Table2 size={12} />{showTable ? 'Show chart' : 'Show table'}
        </button>
      </div>
      {series.length === 0 ? (
        <p className="text-center py-12 text-sm text-slate-600">No checks yet — run a check or inject a fault below</p>
      ) : showTable ? (
        <table className="w-full text-xs">
          <thead><tr className="text-slate-500 text-left"><th className="py-1">Check</th>{STAGES.map((s) => <th key={s} className="py-1 text-right">{STAGE_META[s].short}</th>)}</tr></thead>
          <tbody>
            {series.map((p) => (
              <tr key={p.t} className="border-t border-surface-border/50 font-mono text-slate-300">
                <td className="py-1">{p.t}</td>
                {STAGES.map((s) => <td key={s} className="py-1 text-right">{typeof p[s] === 'number' ? (p[s] as number).toFixed(3) : '—'}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={series} margin={{ top: 8, right: 36, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="#334155" strokeOpacity={0.5} vertical={false} />
            <XAxis dataKey="t" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={{ stroke: '#334155' }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
            <ReferenceLine y={threshold} stroke="#94a3b8" strokeDasharray="4 4" />
            <Tooltip
              cursor={{ stroke: '#64748b', strokeWidth: 1 }}
              contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: '#e2e8f0' }}
              itemStyle={{ color: '#cbd5e1' }}
              formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))}
            />
            <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
            {STAGES.map((s) => (
              <Line key={s} type="monotone" dataKey={s} name={STAGE_META[s].label} stroke={STAGE_META[s].color}
                strokeWidth={2} dot={{ r: 4, strokeWidth: 2, stroke: '#1e293b' }} activeDot={{ r: 6 }}
                isAnimationActive={false} label={endLabel(s)} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

// ─── Attribution panel ───────────────────────────────────────────────────────
function AttributionPanel({ report }: { report: AttributionReport | null }) {
  return (
    <div className="card space-y-4">
      <p className="text-sm font-medium text-slate-300">Root cause attribution</p>
      {!report ? (
        <p className="text-center py-8 text-sm text-slate-600">No drift reported yet</p>
      ) : (
        <>
          <p className="text-sm text-slate-200">{report.explanation}</p>
          <div className="space-y-3">
            {report.ranked_stages.map((r, i) => (
              <div key={r.stage}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="inline-flex items-center gap-2 text-slate-200 font-medium">
                    <span className="text-slate-500 font-mono">#{i + 1}</span><StageSwatch stage={r.stage} />{STAGE_META[r.stage].label}
                    {r.causal_factor < 1 && <span className="badge badge-archived text-[10px]">inherited</span>}
                  </span>
                  <span className="font-mono text-slate-300">{(r.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 rounded bg-surface overflow-hidden">
                  <div className="h-2 rounded" style={{ width: `${r.confidence * 100}%`, background: STAGE_META[r.stage].color }} />
                </div>
                <p className="text-[11px] text-slate-500 mt-1">
                  {r.reason} · magnitude {r.magnitude.toFixed(2)} · onset {fmtTime(r.onset)}
                  {r.onset_lag_seconds > 0 && ` (+${Math.round(r.onset_lag_seconds / 60)} min)`}
                  {r.top_feature && <> · <span className="font-mono">{r.top_feature}</span></>}
                </p>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-slate-600 font-mono">
            score = magnitude × 1/(1 + onset lag/bucket) × causal factor · alert {report.alert_sent_at ? `sent ${fmtTime(report.alert_sent_at)}` : 'not sent'}
          </p>
        </>
      )}
    </div>
  )
}

// ─── Feature tests table ─────────────────────────────────────────────────────
function TestsTable({ stages }: { stages: StageStatus[] }) {
  const rows = stages.flatMap((s) => (s.details?.tests ?? []).slice(0, 4).map((t) => ({ stage: s.stage, ...t })))
  if (!rows.length) return null
  return (
    <div className="card">
      <p className="text-sm font-medium text-slate-300 mb-3">Latest check — strongest signals per stage</p>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 text-left">
            <th className="py-1.5">Stage</th><th>Feature</th><th>Test</th>
            <th className="text-right">Effect</th><th className="text-right">p-value</th><th className="text-right">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-surface-border/50 text-slate-300">
              <td className="py-1.5"><span className="inline-flex items-center gap-1.5"><StageSwatch stage={r.stage} />{STAGE_META[r.stage].short}</span></td>
              <td className="font-mono">{r.feature}</td>
              <td className="text-slate-500">{r.test}</td>
              <td className="text-right font-mono">{r.statistic.toFixed(3)}</td>
              <td className="text-right font-mono">{r.p_value < 0.001 ? '<0.001' : r.p_value.toFixed(3)}</td>
              <td className="text-right">{r.drifted ? <span className="text-drift-critical">drift</span> : <span className="text-slate-500">ok</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Simulate + check controls ───────────────────────────────────────────────
function Controls({ modelId, onDone }: { modelId: string; onDone: (o: CheckOutcome | null) => void }) {
  const [fault, setFault] = useState<FaultType>('s1_shift')
  const [feature, setFeature] = useState('')
  const [magnitude, setMagnitude] = useState(3)

  const { data: model } = useQuery({ queryKey: ['model', modelId], queryFn: () => modelsApi.get(modelId) })
  const simulate = useMutation({
    mutationFn: () => observabilityApi.simulate(modelId, { fault, feature: feature || undefined, magnitude }),
    onSuccess: (r) => onDone(r.check),
  })
  const check = useMutation({ mutationFn: () => observabilityApi.check(modelId), onSuccess: onDone })
  const busy = simulate.isPending || check.isPending
  const err = simulate.error ?? check.error

  return (
    <div className="card space-y-3">
      <p className="text-sm font-medium text-slate-300 inline-flex items-center gap-2"><FlaskConical size={14} />Fault injection (demo)</p>
      <select aria-label="Fault" className="input text-xs" value={fault} onChange={(e) => setFault(e.target.value as FaultType)}>
        {FAULTS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
      </select>
      <p className="text-[11px] text-slate-500">{FAULTS.find((f) => f.value === fault)?.hint}. Generates 600 predictions over the last 30 min; the last half is faulty.</p>
      <div className="grid grid-cols-2 gap-2">
        <select aria-label="Feature" className="input text-xs" value={feature} onChange={(e) => setFeature(e.target.value)} disabled={fault === 's3_concept' || fault === 'none'}>
          <option value="">auto feature</option>
          {(model?.input_columns ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <input aria-label="Magnitude" type="number" min={0.1} step={0.5} className="input text-xs" value={magnitude}
          onChange={(e) => setMagnitude(Number(e.target.value))} disabled={!['s1_shift', 's2_shift'].includes(fault)} />
      </div>
      <div className="flex gap-2">
        <button onClick={() => simulate.mutate()} disabled={busy} className="btn-primary flex-1 justify-center text-sm">
          {simulate.isPending ? <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" /> : <Zap size={14} />}
          Inject & check
        </button>
        <button onClick={() => check.mutate()} disabled={busy} className="btn-secondary justify-center text-sm">
          {check.isPending ? <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" /> : <Play size={14} />}
          Check now
        </button>
      </div>
      {err && <p className="text-xs text-drift-critical">{errorMessage(err)}</p>}
    </div>
  )
}

// ─── Page ────────────────────────────────────────────────────────────────────
export default function ObservabilityDashboard() {
  const queryClient = useQueryClient()
  const [picked, setPicked] = useState<string | null>(null)
  const [lastOutcome, setLastOutcome] = useState<CheckOutcome | null>(null)

  const { data: overview, isLoading } = useQuery({
    queryKey: ['obs-overview'], queryFn: observabilityApi.overview, refetchInterval: 15000,
  })
  // default to the first production model until the user picks one
  const selected = picked ?? overview?.[0]?.model_version_id ?? null

  const { data: drift } = useQuery({
    queryKey: ['obs-drift', selected], queryFn: () => observabilityApi.drift(selected!), enabled: !!selected, refetchInterval: 15000,
  })
  const { data: attribution } = useQuery({
    queryKey: ['obs-attribution', selected], queryFn: () => observabilityApi.attribution(selected!), enabled: !!selected, refetchInterval: 15000,
  })

  const series = useMemo(
    () => (drift?.series ?? []).map((p) => ({ t: new Date(p.detected_at).toLocaleTimeString(), ...p.scores })),
    [drift],
  )
  const stageByName = Object.fromEntries((drift?.latest?.stages ?? []).map((s) => [s.stage, s])) as Partial<Record<Stage, StageStatus>>

  const onDone = (o: CheckOutcome | null) => {
    setLastOutcome(o)
    for (const key of ['obs-overview', 'obs-drift', 'obs-attribution']) queryClient.invalidateQueries({ queryKey: [key] })
  }

  return (
    <div className="animate-slide-up max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="section-title">Observability Dashboard</h1>
        <p className="text-sm text-slate-500 mt-1">
          Independent drift monitors at ingestion, preprocessing and output — ranked by likely root cause
        </p>
      </div>

      {isLoading ? (
        <div className="card text-center py-12 text-slate-600 text-sm">Loading…</div>
      ) : !overview?.length ? (
        <div className="card text-center py-16">
          <Activity size={28} className="text-slate-700 mx-auto mb-3" />
          <p className="text-slate-500 text-sm">Deploy a model to begin monitoring</p>
        </div>
      ) : (
        <>
          <ModelStrip items={overview} selected={selected} onSelect={(id) => { setPicked(id); setLastOutcome(null) }} />

          {lastOutcome && lastOutcome.status !== 'completed' && (
            <div className="px-4 py-3 rounded-lg bg-drift-warning/10 border border-drift-warning/30 text-drift-warning text-sm">
              {lastOutcome.message}
            </div>
          )}

          <div className="grid grid-cols-3 gap-4">
            {STAGES.map((s) => <StageCard key={s} stage={s} status={stageByName[s]} threshold={drift?.threshold ?? 0.1} />)}
          </div>

          <div className="grid grid-cols-5 gap-6">
            <div className="col-span-3 space-y-6">
              <TrendChart series={series} threshold={drift?.threshold ?? 0.1} />
              <TestsTable stages={drift?.latest?.stages ?? []} />
            </div>
            <div className="col-span-2 space-y-6">
              <AttributionPanel report={attribution?.latest ?? null} />
              {selected && <Controls modelId={selected} onDone={onDone} />}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
