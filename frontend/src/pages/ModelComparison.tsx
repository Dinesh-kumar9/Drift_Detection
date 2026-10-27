import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  modelsApi, type ModelStatus, type ModelVersion, type PredictResponse, type RegisteredModel, type SlaTier,
} from '../api/models'
import { errorMessage } from '../api/client'
import { GitCompare, Rocket, Cpu, Clock, TrendingUp, Shield, CheckCircle, AlertTriangle, Layers, Wand2 } from 'lucide-react'
import clsx from 'clsx'

// ─── Status badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    production: 'badge-production',
    staging: 'badge-staging',
    archived: 'badge-archived',
  }
  const icons: Record<string, React.ReactNode> = {
    production: <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse-dot inline-block" />,
    staging: <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />,
    archived: <span className="w-1.5 h-1.5 rounded-full bg-slate-500 inline-block" />,
  }
  return (
    <span className={clsx('badge', map[status] ?? 'badge-staging')}>
      {icons[status]}
      {status}
    </span>
  )
}

// ─── Metric pill ─────────────────────────────────────────────────────────────
function MetricPill({ label, value, unit = '' }: { label: string; value?: number | null; unit?: string }) {
  if (value === undefined || value === null) return null
  return (
    <div className="flex flex-col items-center px-3 py-2 rounded-lg bg-surface border border-surface-border">
      <span className="text-[10px] text-slate-500 uppercase tracking-widest">{label}</span>
      <span className="text-sm font-bold font-mono text-slate-100 mt-0.5">
        {unit === '%' ? (value * 100).toFixed(1) : value.toLocaleString(undefined, { maximumFractionDigits: 4 })}{unit}
      </span>
    </div>
  )
}

// ─── Model version card ──────────────────────────────────────────────────────
function VersionCard({
  model,
  onDeploy,
  deploying,
}: {
  model: ModelVersion
  onDeploy: (id: string, tier: SlaTier) => void
  deploying: boolean
}) {
  const [tier, setTier] = useState<SlaTier>(model.sla_tier ?? 'standard')
  const m = model.metrics ?? {}
  const isProduction = model.status === 'production'
  const isArchived = model.status === 'archived'

  return (
    <div className={clsx(
      'card relative overflow-hidden transition-all duration-300',
      isProduction && 'border-drift-safe/30 shadow-lg shadow-drift-safe/5',
      isArchived && 'opacity-60',
    )}>
      {isProduction && (
        <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-transparent via-drift-safe to-transparent" />
      )}

      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className={clsx(
              'w-8 h-8 rounded-lg flex items-center justify-center',
              isProduction ? 'bg-drift-safe/20' : 'bg-brand-900/60',
            )}>
              <Cpu size={14} className={isProduction ? 'text-drift-safe' : 'text-brand-400'} />
            </div>
            <StatusBadge status={model.status} />
          </div>
          <p className="text-sm font-bold text-slate-100 font-mono mt-1">{model.model_type ?? 'Unknown'}</p>
          <p className="text-[10px] text-slate-600 font-mono mt-0.5">{model.version_tag}</p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-slate-600">SLA</p>
          <span className={clsx(
            'text-xs font-semibold uppercase tracking-wide',
            model.sla_tier === 'critical' ? 'text-drift-critical' :
            model.sla_tier === 'standard' ? 'text-amber-400' : 'text-slate-500',
          )}>
            {model.sla_tier}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-4">
        <MetricPill label="F1" value={m.f1} unit="%" />
        <MetricPill label="Accuracy" value={m.accuracy} unit="%" />
        <MetricPill label="ROC-AUC" value={m.roc_auc} unit="%" />
        <MetricPill label="R²" value={m.r2} />
        <MetricPill label="MAE" value={m.mae} />
        <MetricPill label="Latency" value={m.latency_ms} unit="ms" />
      </div>

      {!isArchived && !isProduction && (
        <div className="flex gap-2">
          <select
            aria-label="SLA tier"
            className="input w-32 text-xs"
            value={tier}
            onChange={(e) => setTier(e.target.value as SlaTier)}
          >
            <option value="critical">critical</option>
            <option value="standard">standard</option>
            <option value="low">low</option>
          </select>
          <button
            onClick={() => onDeploy(model.id, tier)}
            disabled={deploying}
            className="btn-primary flex-1 justify-center text-sm"
          >
            {deploying ? (
              <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Rocket size={14} />
            )}
            Deploy to Production
          </button>
        </div>
      )}
      {isProduction && (
        <div className="flex items-center gap-2 text-sm text-drift-safe font-medium">
          <CheckCircle size={14} />
          Live in Production
          {model.deployed_at && (
            <span className="text-[10px] text-slate-500 font-mono ml-auto">
              {new Date(model.deployed_at).toLocaleString()}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Predict tester ───────────────────────────────────────────────────────────
function PredictTester({ productionModels }: { productionModels: ModelVersion[] }) {
  const [selectedId, setSelectedId] = useState<string>('')
  const [featuresRaw, setFeaturesRaw] = useState('{}')
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const model = productionModels.find((m) => m.id === selectedId) ?? productionModels[0]

  const sampleMutation = useMutation({
    mutationFn: (id: string) => modelsApi.sampleInput(id),
    onSuccess: (features) => { setFeaturesRaw(JSON.stringify(features, null, 2)); setError(null) },
    onError: (e) => setError(errorMessage(e, 'Could not load a sample input')),
  })

  const predictMutation = useMutation({
    mutationFn: async () => {
      if (!model) throw new Error('No production model')
      let features: Record<string, unknown>
      try {
        features = JSON.parse(featuresRaw)
      } catch {
        throw new Error('Feature JSON is not valid JSON')
      }
      return modelsApi.predict(model.id, features)
    },
    onSuccess: (data) => { setResult(data); setError(null) },
    onError: (e) => { setError(errorMessage(e, 'Inference failed')); setResult(null) },
  })

  if (!model) {
    return (
      <div className="card text-center py-8 text-slate-700 text-sm">
        Deploy a model to enable inference testing
      </div>
    )
  }

  return (
    <div className="card space-y-4">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Live Inference Tester</p>

      <select
        aria-label="Production model"
        className="input text-xs"
        value={model.id}
        onChange={(e) => { setSelectedId(e.target.value); setResult(null); setError(null); setFeaturesRaw('{}') }}
      >
        {productionModels.map((m) => (
          <option key={m.id} value={m.id}>
            {m.registered_model_name} · {m.model_type} {m.version_tag}
          </option>
        ))}
      </select>

      <div className="flex items-center gap-2 text-xs text-drift-safe">
        <span className="w-2 h-2 rounded-full bg-drift-safe animate-pulse-dot" />
        predicts <span className="font-mono">{model.target_column}</span> ({model.task_type})
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs text-slate-500">Feature JSON</p>
          <button
            onClick={() => sampleMutation.mutate(model.id)}
            disabled={sampleMutation.isPending}
            className="text-[11px] text-brand-400 hover:text-brand-300 inline-flex items-center gap-1"
          >
            <Wand2 size={11} /> Fill sample from training data
          </button>
        </div>
        <textarea
          aria-label="Feature JSON"
          className="input font-mono text-xs h-40 resize-none"
          value={featuresRaw}
          onChange={(e) => setFeaturesRaw(e.target.value)}
        />
        {model.input_columns && (
          <p className="text-[10px] text-slate-600 mt-1 font-mono break-words">
            required: {model.input_columns.join(', ')}
          </p>
        )}
      </div>

      <button
        onClick={() => predictMutation.mutate()}
        disabled={predictMutation.isPending}
        className="btn-primary w-full justify-center text-sm"
      >
        {predictMutation.isPending
          ? <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          : <TrendingUp size={14} />
        }
        Run Inference
      </button>

      {result && (
        <div className="p-3 rounded-lg bg-drift-safe/10 border border-drift-safe/30 animate-fade-in space-y-2">
          <div className="flex items-baseline justify-between">
            <p className="text-xs text-drift-safe font-semibold">Prediction</p>
            <p className="text-[10px] text-slate-500 font-mono">{result.latency_ms.toFixed(1)} ms</p>
          </div>
          <p className="text-lg font-bold font-mono text-slate-100">{String(result.prediction)}</p>
          {result.probabilities && (
            <div className="space-y-1">
              {Object.entries(result.probabilities).map(([label, p]) => (
                <div key={label} className="flex items-center gap-2 text-[11px] font-mono">
                  <span className="w-20 truncate text-slate-400">{label}</span>
                  <div className="flex-1 h-1.5 bg-surface rounded-full overflow-hidden">
                    <div className="h-full bg-drift-safe rounded-full" style={{ width: `${p * 100}%` }} />
                  </div>
                  <span className="w-12 text-right text-slate-300">{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
          <p className="text-[10px] text-slate-600 font-mono">logged as {result.prediction_id}</p>
        </div>
      )}
      {error && (
        <div className="p-3 rounded-lg bg-drift-critical/10 border border-drift-critical/30">
          <p className="text-xs text-drift-critical break-words">{error}</p>
        </div>
      )}
    </div>
  )
}

// ─── Lineage section ─────────────────────────────────────────────────────────
function LineageSection({
  registered,
  versions,
  onDeploy,
  deployingId,
}: {
  registered: RegisteredModel | undefined
  versions: ModelVersion[]
  onDeploy: (id: string, tier: SlaTier) => void
  deployingId: string | null
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <Layers size={14} className="text-brand-400" />
        <p className="text-sm font-semibold text-slate-200 font-mono">{registered?.name ?? 'Unregistered'}</p>
        {registered && (
          <p className="text-[11px] text-slate-500">
            {registered.task_type} · target <span className="font-mono">{registered.target_column}</span> ·{' '}
            {registered.version_count} version{registered.version_count === 1 ? '' : 's'}
          </p>
        )}
      </div>
      {versions.map((model) => (
        <VersionCard key={model.id} model={model} onDeploy={onDeploy} deploying={deployingId === model.id} />
      ))}
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function ModelComparison() {
  const queryClient = useQueryClient()
  const [statusFilter, setStatusFilter] = useState<ModelStatus | undefined>(undefined)

  const { data: models, isLoading } = useQuery({
    queryKey: ['models', statusFilter],
    queryFn: () => modelsApi.list({ status: statusFilter }),
    refetchInterval: 10000,
  })
  const { data: registered } = useQuery({
    queryKey: ['registered-models'],
    queryFn: modelsApi.registered,
    refetchInterval: 10000,
  })
  const { data: production } = useQuery({
    queryKey: ['models', 'production'],
    queryFn: () => modelsApi.list({ status: 'production' }),
    refetchInterval: 10000,
  })

  const deployMutation = useMutation({
    mutationFn: ({ id, tier }: { id: string; tier: SlaTier }) => modelsApi.deploy(id, tier),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
      queryClient.invalidateQueries({ queryKey: ['registered-models'] })
    },
  })

  const lineages = useMemo(() => {
    const byId = new Map<string, ModelVersion[]>()
    for (const m of models?.items ?? []) {
      const key = m.registered_model_id ?? 'none'
      byId.set(key, [...(byId.get(key) ?? []), m])
    }
    return [...byId.entries()]
  }, [models])

  const registeredById = new Map((registered ?? []).map((r) => [r.id, r]))
  const items = models?.items ?? []

  return (
    <div className="animate-slide-up max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="section-title">Model Registry</h1>
          <p className="text-sm text-slate-500 mt-1">
            Each named model keeps its own production version — promote, compare and run live inference
          </p>
        </div>
        <div className="flex gap-2">
          {(['All', 'staging', 'production', 'archived'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setStatusFilter(f === 'All' ? undefined : f)}
              className={clsx(
                'px-3 py-1.5 text-xs rounded-lg font-medium transition-all duration-200',
                (f === 'All' && !statusFilter) || statusFilter === f
                  ? 'bg-brand-600 text-white'
                  : 'bg-surface-card text-slate-400 hover:text-slate-200 border border-surface-border',
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Registered Models', count: registered?.length ?? 0, icon: Layers, color: 'text-brand-400' },
          { label: 'In Production', count: production?.total ?? 0, icon: Shield, color: 'text-drift-safe' },
          {
            label: statusFilter ? `Shown (${statusFilter})` : 'Versions',
            count: models?.total ?? 0,
            icon: statusFilter === 'archived' ? Clock : GitCompare,
            color: 'text-amber-400',
          },
        ].map(({ label, count, icon: Icon, color }) => (
          <div key={label} className="card flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-surface flex items-center justify-center">
              <Icon size={18} className={color} />
            </div>
            <div>
              <p className="metric-label">{label}</p>
              <p className="metric-value text-2xl">{count}</p>
            </div>
          </div>
        ))}
      </div>

      {deployMutation.isError && (
        <div className="px-4 py-3 rounded-lg bg-drift-critical/10 border border-drift-critical/30 text-drift-critical text-sm">
          {errorMessage(deployMutation.error, 'Deploy failed')}
        </div>
      )}

      <div className="grid grid-cols-5 gap-6">
        <div className="col-span-3 space-y-6">
          {isLoading ? (
            <div className="card text-center py-12 text-slate-600 text-sm">Loading models…</div>
          ) : items.length === 0 ? (
            <div className="card text-center py-16">
              <AlertTriangle size={28} className="text-slate-700 mx-auto mb-3" />
              <p className="text-slate-500 text-sm">No models yet — train some via the Experiment Dashboard</p>
            </div>
          ) : (
            lineages.map(([rmId, versions]) => (
              <LineageSection
                key={rmId}
                registered={registeredById.get(rmId)}
                versions={versions}
                onDeploy={(id, tier) => deployMutation.mutate({ id, tier })}
                deployingId={deployMutation.isPending ? deployMutation.variables?.id ?? null : null}
              />
            ))
          )}
        </div>

        <div className="col-span-2">
          <PredictTester productionModels={production?.items ?? []} />
        </div>
      </div>
    </div>
  )
}
