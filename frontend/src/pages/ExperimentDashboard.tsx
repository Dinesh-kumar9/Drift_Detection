import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { experimentsApi } from '../api/experiments'
import { datasetsApi } from '../api/datasets'
import { Play, Clock, CheckCircle, XCircle, RefreshCw, ChevronRight } from 'lucide-react'
import clsx from 'clsx'

// ─── Status badge ─────────────────────────────────────────────────────────────
function StatusPill({ status }: { status: string }) {
  const map: Record<string, string> = {
    completed: 'badge-safe',
    running: 'bg-brand-500/20 text-brand-300 ring-1 ring-brand-500/30',
    queued: 'bg-slate-600/20 text-slate-400 ring-1 ring-slate-500/30',
    failed: 'badge-critical',
  }
  const icons: Record<string, React.ReactNode> = {
    completed: <CheckCircle size={10} />,
    running: <RefreshCw size={10} className="animate-spin" />,
    queued: <Clock size={10} />,
    failed: <XCircle size={10} />,
  }
  return (
    <span className={clsx('badge', map[status] || 'badge-staging')}>
      {icons[status]}
      {status}
    </span>
  )
}

// ─── Metric bar ───────────────────────────────────────────────────────────────
function MetricBar({ label, value, max = 1, highlight }: {
  label: string; value: number | undefined; max?: number; highlight?: boolean
}) {
  if (value === undefined) return null
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs text-slate-500">{label}</span>
        <span className={clsx('text-xs font-mono font-bold', highlight ? 'text-drift-safe' : 'text-slate-300')}>
          {(value * (max === 1 ? 100 : 1)).toFixed(max === 1 ? 1 : 2)}{max === 1 ? '%' : ''}
        </span>
      </div>
      <div className="h-1.5 bg-surface rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full transition-all duration-700', highlight ? 'bg-drift-safe' : 'bg-brand-500')}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ─── Training trigger form ────────────────────────────────────────────────────
function TrainForm({ onSubmit }: { onSubmit: (datasetId: string, targetCol: string) => void }) {
  const [datasetId, setDatasetId] = useState('')
  const [targetCol, setTargetCol] = useState('')

  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetsApi.list(50, 0),
  })

  const selectedDataset = datasets?.items.find((d) => d.id === datasetId)
  const columns = selectedDataset?.schema_json ? Object.keys(selectedDataset.schema_json) : []

  return (
    <div className="card space-y-4">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">New Training Run</p>

      <div>
        <label className="text-xs text-slate-500 mb-1.5 block">Dataset</label>
        <select
          className="input"
          value={datasetId}
          onChange={(e) => { setDatasetId(e.target.value); setTargetCol('') }}
        >
          <option value="">Select a dataset…</option>
          {datasets?.items.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name} ({d.row_count ? parseInt(d.row_count).toLocaleString() : '?'} rows)
            </option>
          ))}
        </select>
      </div>

      {datasetId && (
        <div className="animate-slide-up">
          <label className="text-xs text-slate-500 mb-1.5 block">Target Column</label>
          <select
            className="input"
            value={targetCol}
            onChange={(e) => setTargetCol(e.target.value)}
          >
            <option value="">Select target column…</option>
            {columns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      )}

      <button
        onClick={() => onSubmit(datasetId, targetCol)}
        disabled={!datasetId || !targetCol}
        className="btn-primary w-full justify-center"
      >
        <Play size={14} />
        Train 3 Models in Parallel
      </button>

      <p className="text-[10px] text-slate-600 text-center">
        RandomForest · XGBoost · LogReg — async via Celery
      </p>
    </div>
  )
}

// ─── Comparison panel ─────────────────────────────────────────────────────────
function ComparisonPanel({
  groupId,
  onDeploy,
}: {
  groupId: string
  onDeploy: (runId: string) => void
}) {
  // Poll status
  const { data: status } = useQuery({
    queryKey: ['exp-status', groupId],
    queryFn: () => experimentsApi.status(groupId),
    refetchInterval: (q) =>
      q.state.data?.overall_status === 'completed' || q.state.data?.overall_status === 'failed' ? false : 2000,
    enabled: !!groupId,
  })

  // Fetch comparison when done
  const { data: comparison } = useQuery({
    queryKey: ['exp-compare', groupId],
    queryFn: () => experimentsApi.compare(groupId),
    enabled: status?.overall_status === 'completed',
  })

  const isClassification = comparison?.task_type === 'classification'

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Status bar */}
      <div className="card flex items-center justify-between">
        <div>
          <p className="text-xs text-slate-500 font-mono">Group ID</p>
          <p className="text-xs text-slate-400 font-mono truncate max-w-[24ch]">{groupId}</p>
        </div>
        <StatusPill status={status?.overall_status ?? 'queued'} />
      </div>

      {/* Per-model status tiles */}
      {status && (
        <div className="grid grid-cols-3 gap-3">
          {status.models.map((m) => (
            <div key={m.run_id} className="card py-3 text-center">
              <p className="text-xs font-mono text-slate-500 truncate">{m.model_type}</p>
              <div className="mt-2 flex justify-center">
                <StatusPill status={m.status} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Comparison metrics */}
      {comparison && (
        <div className="space-y-3 animate-slide-up">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Model Comparison</p>
          {comparison.models.map((m) => {
            const isBest = m.run_id === comparison.best_model_run_id
            return (
              <div key={m.run_id} className={clsx(
                'card relative overflow-hidden',
                isBest && 'border-drift-safe/40',
              )}>
                {isBest && (
                  <div className="absolute top-0 right-0 text-[9px] bg-drift-safe text-black font-bold px-2 py-0.5 rounded-bl-lg">
                    BEST
                  </div>
                )}
                <div className="flex items-center justify-between mb-3">
                  <p className="text-sm font-semibold text-slate-200 font-mono">{m.model_type}</p>
                  <StatusPill status={m.status} />
                </div>
                <div className="space-y-2">
                  {isClassification ? (
                    <>
                      <MetricBar label="F1 Score" value={m.f1} highlight={isBest} />
                      <MetricBar label="Accuracy" value={m.accuracy} />
                      <MetricBar label="ROC-AUC" value={m.roc_auc} />
                    </>
                  ) : (
                    <>
                      <MetricBar label="R²" value={m.r2} highlight={isBest} />
                      <MetricBar label="MAE" value={m.mae} max={100} />
                    </>
                  )}
                </div>
                <div className="mt-3 flex items-center justify-between text-[10px] text-slate-600 font-mono">
                  <span>Training: {m.training_time_s?.toFixed(1)}s</span>
                  <span>Latency: {m.latency_ms?.toFixed(3)}ms/sample</span>
                </div>
                {isBest && (
                  <button
                    onClick={() => onDeploy(m.run_id)}
                    className="btn-primary mt-3 w-full justify-center text-sm"
                  >
                    <ChevronRight size={14} />
                    Deploy to Staging
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function ExperimentDashboard() {
  const queryClient = useQueryClient()
  const [activeGroupId, setActiveGroupId] = useState<string | null>(null)

  const { data: allRuns, isLoading } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => experimentsApi.list(undefined, 50),
    refetchInterval: 10000,
  })

  const trainMutation = useMutation({
    mutationFn: ({ datasetId, targetCol }: { datasetId: string; targetCol: string }) =>
      experimentsApi.train({ dataset_id: datasetId, target_column: targetCol }),
    onSuccess: (res) => {
      setActiveGroupId(res.experiment_group_id)
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
    },
  })

  const completedGroups = allRuns?.items.filter((r) => r.status === 'completed') ?? []
  const runningCount = allRuns?.items.filter((r) => r.status === 'running' || r.status === 'queued').length ?? 0

  return (
    <div className="animate-slide-up max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="section-title">Experiment Dashboard</h1>
          <p className="text-sm text-slate-500 mt-1">Trigger parallel multi-model training and compare results</p>
        </div>
        <div className="flex gap-3">
          {[
            { label: 'Total Runs', value: allRuns?.total ?? 0 },
            { label: 'Completed', value: completedGroups.length },
            { label: 'In Progress', value: runningCount },
          ].map(({ label, value }) => (
            <div key={label} className="card py-3 px-5 text-center min-w-[80px]">
              <p className="metric-value text-2xl">{value}</p>
              <p className="metric-label mt-0.5">{label}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-5 gap-6">
        {/* Left: train form + history */}
        <div className="col-span-2 space-y-4">
          <TrainForm
            onSubmit={(datasetId, targetCol) => trainMutation.mutate({ datasetId, targetCol })}
          />
          {trainMutation.isPending && (
            <div className="card flex items-center gap-3 text-sm text-slate-400 animate-fade-in">
              <div className="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
              Dispatching training job to Celery…
            </div>
          )}

          {/* History */}
          <div className="card">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">Recent Runs</p>
            {isLoading ? (
              <div className="text-center py-6 text-slate-600 text-sm">Loading…</div>
            ) : (allRuns?.items ?? []).filter(r => r.model_type !== '__group__').length === 0 ? (
              <div className="text-center py-6 text-slate-700 text-sm">No runs yet</div>
            ) : (
              <div className="space-y-1.5 max-h-72 overflow-y-auto">
                {(allRuns?.items ?? []).filter(r => r.model_type !== '__group__').slice(0, 15).map((r) => (
                  <div key={r.id} className="flex items-center justify-between px-3 py-2 rounded-lg hover:bg-surface-border/40 transition-colors">
                    <div>
                      <p className="text-xs font-mono text-slate-300">{r.model_type}</p>
                      <p className="text-[10px] text-slate-600">{new Date(r.created_at).toLocaleString()}</p>
                    </div>
                    <StatusPill status={r.status} />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: live comparison */}
        <div className="col-span-3">
          {activeGroupId ? (
            <ComparisonPanel
              groupId={activeGroupId}
              onDeploy={(runId) => {
                // navigate to models tab — for now alert
                alert(`Run ${runId} ready to deploy — go to Model Comparison page!`)
              }}
            />
          ) : (
            <div className="card h-full flex flex-col items-center justify-center py-24 text-center">
              <Play size={36} className="text-slate-700 mb-4" />
              <p className="text-slate-500 text-sm">Trigger a training run to see live comparison</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
