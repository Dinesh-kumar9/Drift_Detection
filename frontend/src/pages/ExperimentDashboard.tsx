import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { experimentsApi, type MetricSet, type RunState, type TaskType, type TrainRequest } from '../api/experiments'
import { datasetsApi } from '../api/datasets'
import { modelsApi } from '../api/models'
import { errorMessage } from '../api/client'
import { Play, Clock, CheckCircle, XCircle, RefreshCw, Rocket, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

// ─── Status badge ─────────────────────────────────────────────────────────────
function StatusPill({ status }: { status: RunState | string }) {
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

// ─── Metric bar (0–1 metrics) ─────────────────────────────────────────────────
function MetricBar({ label, value, highlight }: { label: string; value?: number | null; highlight?: boolean }) {
  if (value === undefined || value === null) return null
  const pct = Math.max(0, Math.min(value * 100, 100))
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs text-slate-500">{label}</span>
        <span className={clsx('text-xs font-mono font-bold', highlight ? 'text-drift-safe' : 'text-slate-300')}>
          {(value * 100).toFixed(1)}%
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

function MetricText({ label, value }: { label: string; value?: number | null }) {
  if (value === undefined || value === null) return null
  return (
    <div className="flex justify-between text-xs">
      <span className="text-slate-500">{label}</span>
      <span className="font-mono text-slate-300">{value.toLocaleString(undefined, { maximumFractionDigits: 4 })}</span>
    </div>
  )
}

// ─── Training trigger form ────────────────────────────────────────────────────
function TrainForm({ onSubmit, pending }: { onSubmit: (req: TrainRequest) => void; pending: boolean }) {
  const [datasetId, setDatasetId] = useState('')
  const [targetCol, setTargetCol] = useState('')
  const [taskType, setTaskType] = useState<TaskType>('auto')
  const [modelName, setModelName] = useState('')

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
        <label htmlFor="dataset" className="text-xs text-slate-500 mb-1.5 block">Dataset</label>
        <select
          id="dataset"
          className="input"
          value={datasetId}
          onChange={(e) => { setDatasetId(e.target.value); setTargetCol('') }}
        >
          <option value="">Select a dataset…</option>
          {datasets?.items.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name} ({d.row_count != null ? d.row_count.toLocaleString() : '?'} rows)
            </option>
          ))}
        </select>
      </div>

      {datasetId && (
        <div className="space-y-4 animate-slide-up">
          <div>
            <label htmlFor="target" className="text-xs text-slate-500 mb-1.5 block">Target Column</label>
            <select id="target" className="input" value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
              <option value="">Select target column…</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="task" className="text-xs text-slate-500 mb-1.5 block">Task</label>
              <select id="task" className="input" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}>
                <option value="auto">Auto-detect</option>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>
            <div>
              <label htmlFor="model-name" className="text-xs text-slate-500 mb-1.5 block">Model name</label>
              <input
                id="model-name"
                className="input"
                placeholder="auto"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
              />
            </div>
          </div>
        </div>
      )}

      <button
        onClick={() =>
          onSubmit({
            dataset_id: datasetId,
            target_column: targetCol,
            task_type: taskType,
            model_name: modelName.trim() || undefined,
          })
        }
        disabled={!datasetId || !targetCol || pending}
        className="btn-primary w-full justify-center"
      >
        {pending
          ? <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          : <Play size={14} />}
        Train 3 Models in Parallel
      </button>

      <p className="text-[10px] text-slate-600 text-center">
        RandomForest · XGBoost · Logistic/Ridge — async via Celery. Versions are added to the named model.
      </p>
    </div>
  )
}

// ─── Comparison panel ─────────────────────────────────────────────────────────
function ModelResultCard({
  m,
  isBest,
  isClassification,
  onDeploy,
  deploying,
}: {
  m: MetricSet
  isBest: boolean
  isClassification: boolean
  onDeploy: (modelVersionId: string) => void
  deploying: boolean
}) {
  return (
    <div className={clsx('card relative overflow-hidden', isBest && 'border-drift-safe/40')}>
      {isBest && (
        <div className="absolute top-0 right-0 text-[9px] bg-drift-safe text-black font-bold px-2 py-0.5 rounded-bl-lg">
          BEST
        </div>
      )}
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm font-semibold text-slate-200 font-mono">
          {m.model_type}
          {m.version_tag && <span className="ml-2 text-[10px] text-slate-500">{m.version_tag}</span>}
        </p>
        <StatusPill status={m.status} />
      </div>

      {m.status === 'failed' ? (
        <p className="text-xs text-drift-critical font-mono break-words">{m.error ?? 'Training failed'}</p>
      ) : (
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
              <MetricText label="MAE" value={m.mae} />
              <MetricText label="RMSE" value={m.rmse} />
            </>
          )}
        </div>
      )}

      {m.status === 'completed' && (
        <>
          <div className="mt-3 flex items-center justify-between text-[10px] text-slate-600 font-mono">
            <span>Train: {m.training_time_s?.toFixed(1)}s</span>
            <span>{m.latency_ms?.toFixed(3)}ms/sample</span>
            {m.model_size_kb != null && <span>{m.model_size_kb.toLocaleString()} KB</span>}
          </div>
          {m.model_version_id && (
            <button
              onClick={() => onDeploy(m.model_version_id!)}
              disabled={deploying}
              className={clsx(isBest ? 'btn-primary' : 'btn-secondary', 'mt-3 w-full justify-center text-sm')}
            >
              {deploying
                ? <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                : <Rocket size={14} />}
              Deploy to Production
            </button>
          )}
        </>
      )}
    </div>
  )
}

function ComparisonPanel({ groupId }: { groupId: string }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: status } = useQuery({
    queryKey: ['exp-status', groupId],
    queryFn: () => experimentsApi.status(groupId),
    refetchInterval: (q) =>
      q.state.data?.overall_status === 'completed' || q.state.data?.overall_status === 'failed' ? false : 2000,
  })

  const finished = status?.overall_status === 'completed' || status?.overall_status === 'failed'
  const { data: comparison } = useQuery({
    queryKey: ['exp-compare', groupId],
    queryFn: () => experimentsApi.compare(groupId),
    enabled: finished,
  })

  const deployMutation = useMutation({
    mutationFn: (modelVersionId: string) => modelsApi.deploy(modelVersionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
      queryClient.invalidateQueries({ queryKey: ['registered-models'] })
      navigate('/models')
    },
  })

  const isClassification = comparison?.task_type === 'classification'

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="card flex items-center justify-between">
        <div>
          <p className="text-xs text-slate-500 font-mono">Group ID</p>
          <p className="text-xs text-slate-400 font-mono truncate max-w-[36ch]">{groupId}</p>
        </div>
        <StatusPill status={status?.overall_status ?? 'queued'} />
      </div>

      {status?.error && (
        <div className="flex items-start gap-2 px-4 py-3 rounded-lg bg-drift-critical/10 border border-drift-critical/30 text-drift-critical text-xs">
          <AlertCircle size={14} className="flex-shrink-0 mt-0.5" />
          <span className="break-words">{status.error}</span>
        </div>
      )}
      {deployMutation.isError && (
        <div className="px-4 py-3 rounded-lg bg-drift-critical/10 border border-drift-critical/30 text-drift-critical text-xs">
          {errorMessage(deployMutation.error, 'Deploy failed')}
        </div>
      )}

      {status && !comparison && (
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

      {comparison && (
        <div className="space-y-3 animate-slide-up">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
            Model Comparison · ranked by {comparison.best_metric}
          </p>
          {comparison.models.map((m) => (
            <ModelResultCard
              key={m.run_id}
              m={m}
              isBest={m.run_id === comparison.best_model_run_id}
              isClassification={isClassification}
              onDeploy={(id) => deployMutation.mutate(id)}
              deploying={deployMutation.isPending && deployMutation.variables === m.model_version_id}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function ExperimentDashboard() {
  const queryClient = useQueryClient()
  const [activeGroupId, setActiveGroupId] = useState<string | null>(null)

  const { data: groups, isLoading } = useQuery({
    queryKey: ['experiment-groups'],
    queryFn: () => experimentsApi.groups(30),
    refetchInterval: 5000,
  })

  const trainMutation = useMutation({
    mutationFn: (req: TrainRequest) => experimentsApi.train(req),
    onSuccess: (res) => {
      setActiveGroupId(res.experiment_group_id)
      queryClient.invalidateQueries({ queryKey: ['experiment-groups'] })
    },
  })

  const items = groups?.items ?? []
  const completed = items.filter((g) => g.status === 'completed').length
  const inProgress = items.filter((g) => g.status === 'running' || g.status === 'queued').length

  return (
    <div className="animate-slide-up max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="section-title">Experiment Dashboard</h1>
          <p className="text-sm text-slate-500 mt-1">Trigger parallel multi-model training and compare results</p>
        </div>
        <div className="flex gap-3">
          {[
            { label: 'Experiments', value: groups?.total ?? 0 },
            { label: 'Completed', value: completed },
            { label: 'In Progress', value: inProgress },
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
          <TrainForm onSubmit={(req) => trainMutation.mutate(req)} pending={trainMutation.isPending} />
          {trainMutation.isError && (
            <div className="px-4 py-3 rounded-lg bg-drift-critical/10 border border-drift-critical/30 text-drift-critical text-xs animate-fade-in">
              {errorMessage(trainMutation.error, 'Could not start training')}
            </div>
          )}

          <div className="card">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">Recent Experiments</p>
            {isLoading ? (
              <div className="text-center py-6 text-slate-600 text-sm">Loading…</div>
            ) : items.length === 0 ? (
              <div className="text-center py-6 text-slate-700 text-sm">No experiments yet</div>
            ) : (
              <div className="space-y-1.5 max-h-80 overflow-y-auto">
                {items.map((g) => (
                  <button
                    key={g.id}
                    onClick={() => setActiveGroupId(g.id)}
                    className={clsx(
                      'w-full text-left flex items-center justify-between gap-2 px-3 py-2 rounded-lg transition-colors border',
                      activeGroupId === g.id
                        ? 'bg-brand-600/10 border-brand-500/30'
                        : 'border-transparent hover:bg-surface-border/40',
                    )}
                  >
                    <div className="min-w-0">
                      <p className="text-xs font-mono text-slate-300 truncate">
                        {g.registered_model_name ?? g.dataset_name}
                      </p>
                      <p className="text-[10px] text-slate-600 truncate">
                        {g.dataset_name} → {g.target_column} · {g.resolved_task_type ?? g.task_type} ·{' '}
                        {new Date(g.created_at).toLocaleString()}
                      </p>
                    </div>
                    <StatusPill status={g.status} />
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: live comparison */}
        <div className="col-span-3">
          {activeGroupId ? (
            <ComparisonPanel key={activeGroupId} groupId={activeGroupId} />
          ) : (
            <div className="card h-full flex flex-col items-center justify-center py-24 text-center">
              <Play size={36} className="text-slate-700 mb-4" />
              <p className="text-slate-500 text-sm">Trigger a training run, or pick a recent experiment to compare</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
