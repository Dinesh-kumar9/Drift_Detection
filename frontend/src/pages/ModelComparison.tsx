import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { modelsApi, type ModelVersion } from '../api/models'
import { GitCompare, Rocket, Cpu, Clock, TrendingUp, Shield, CheckCircle, AlertTriangle } from 'lucide-react'
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
function MetricPill({ label, value, unit = '' }: { label: string; value?: number; unit?: string }) {
  if (value === undefined) return null
  return (
    <div className="flex flex-col items-center px-3 py-2 rounded-lg bg-surface border border-surface-border">
      <span className="text-[10px] text-slate-500 uppercase tracking-widest">{label}</span>
      <span className="text-sm font-bold font-mono text-slate-100 mt-0.5">
        {unit === '%' ? (value * 100).toFixed(1) : value.toFixed(4)}{unit}
      </span>
    </div>
  )
}

// ─── Model card ──────────────────────────────────────────────────────────────
function ModelCard({
  model,
  onDeploy,
  deploying,
}: {
  model: ModelVersion
  onDeploy: (id: string) => void
  deploying: boolean
}) {
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

      {/* Header */}
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
          <p className="text-sm font-bold text-slate-100 font-mono mt-1">
            {model.model_type ?? 'Unknown'}
          </p>
          <p className="text-[10px] text-slate-600 font-mono mt-0.5">
            {model.version_tag ?? 'v1.0.0'}
          </p>
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

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        {m.f1 !== undefined && <MetricPill label="F1" value={m.f1} unit="%" />}
        {m.accuracy !== undefined && <MetricPill label="Accuracy" value={m.accuracy} unit="%" />}
        {m.roc_auc !== undefined && <MetricPill label="ROC-AUC" value={m.roc_auc} unit="%" />}
        {m.latency_ms !== undefined && <MetricPill label="Latency" value={m.latency_ms} unit="ms" />}
        {m.r2 !== undefined && <MetricPill label="R²" value={m.r2} />}
        {m.mae !== undefined && <MetricPill label="MAE" value={m.mae} />}
      </div>

      {/* Deploy button */}
      {!isArchived && !isProduction && (
        <button
          onClick={() => onDeploy(model.id)}
          disabled={deploying}
          className="btn-primary w-full justify-center text-sm"
        >
          {deploying ? (
            <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <Rocket size={14} />
          )}
          Deploy to Production
        </button>
      )}
      {isProduction && (
        <div className="flex items-center gap-2 text-sm text-drift-safe font-medium">
          <CheckCircle size={14} />
          Live in Production
          {model.deployed_at && (
            <span className="text-[10px] text-slate-500 font-mono ml-auto">
              {new Date(model.deployed_at).toLocaleDateString()}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Predict tester ───────────────────────────────────────────────────────────
function PredictTester({ productionModel }: { productionModel: ModelVersion | null }) {
  const [featuresRaw, setFeaturesRaw] = useState('{\n  "V1": -1.359807,\n  "V2": -0.072781,\n  "Amount": 149.62,\n  "Class": 0\n}')
  const [result, setResult] = useState<unknown>(null)
  const [error, setError] = useState<string | null>(null)

  const predictMutation = useMutation({
    mutationFn: async () => {
      if (!productionModel) throw new Error('No production model')
      const features = JSON.parse(featuresRaw)
      return modelsApi.predict(productionModel.id, features)
    },
    onSuccess: (data) => { setResult(data); setError(null) },
    onError: (e: Error) => { setError(e.message); setResult(null) },
  })

  return (
    <div className="card space-y-4">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Live Inference Tester</p>
      {productionModel ? (
        <>
          <div className="flex items-center gap-2 text-xs text-drift-safe">
            <span className="w-2 h-2 rounded-full bg-drift-safe animate-pulse-dot" />
            {productionModel.model_type} — {productionModel.version_tag}
          </div>
          <div>
            <p className="text-xs text-slate-500 mb-2">Feature JSON</p>
            <textarea
              className="input font-mono text-xs h-32 resize-none"
              value={featuresRaw}
              onChange={(e) => setFeaturesRaw(e.target.value)}
            />
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
            <div className="p-3 rounded-lg bg-drift-safe/10 border border-drift-safe/30 animate-fade-in">
              <p className="text-xs text-drift-safe font-semibold mb-2">Prediction</p>
              <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
          {error && (
            <div className="p-3 rounded-lg bg-drift-critical/10 border border-drift-critical/30">
              <p className="text-xs text-drift-critical">{error}</p>
            </div>
          )}
        </>
      ) : (
        <div className="text-center py-8 text-slate-700 text-sm">
          Deploy a model to enable inference testing
        </div>
      )}
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function ModelComparison() {
  const queryClient = useQueryClient()
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined)

  const { data: models, isLoading } = useQuery({
    queryKey: ['models', statusFilter],
    queryFn: () => modelsApi.list(statusFilter),
    refetchInterval: 10000,
  })

  const { data: productionModel } = useQuery({
    queryKey: ['models-production'],
    queryFn: modelsApi.getProduction,
    retry: false,
  })

  const [deployingId, setDeployingId] = useState<string | null>(null)
  const deployMutation = useMutation({
    mutationFn: (id: string) => modelsApi.deploy(id, 'standard'),
    onMutate: (id) => setDeployingId(id),
    onSettled: () => setDeployingId(null),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
      queryClient.invalidateQueries({ queryKey: ['models-production'] })
    },
  })

  const production = models?.items.filter((m) => m.status === 'production') ?? []
  const staging = models?.items.filter((m) => m.status === 'staging') ?? []
  const archived = models?.items.filter((m) => m.status === 'archived') ?? []

  return (
    <div className="animate-slide-up max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="section-title">Model Comparison & Registry</h1>
          <p className="text-sm text-slate-500 mt-1">
            Compare trained models, promote to production, and run live inference
          </p>
        </div>
        <div className="flex gap-2">
          {['All', 'staging', 'production', 'archived'].map((f) => (
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

      {/* Summary counts */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Production', count: production.length, icon: Shield, color: 'text-drift-safe' },
          { label: 'Staging', count: staging.length, icon: GitCompare, color: 'text-amber-400' },
          { label: 'Archived', count: archived.length, icon: Clock, color: 'text-slate-500' },
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

      {/* Main grid */}
      <div className="grid grid-cols-5 gap-6">
        {/* Models */}
        <div className="col-span-3 space-y-4">
          {isLoading ? (
            <div className="card text-center py-12 text-slate-600 text-sm">Loading models…</div>
          ) : models?.items.length === 0 ? (
            <div className="card text-center py-16">
              <AlertTriangle size={28} className="text-slate-700 mx-auto mb-3" />
              <p className="text-slate-500 text-sm">No models yet — train some via the Experiment Dashboard</p>
            </div>
          ) : (
            <div className="space-y-3">
              {models?.items.map((model) => (
                <ModelCard
                  key={model.id}
                  model={model}
                  onDeploy={(id) => deployMutation.mutate(id)}
                  deploying={deployingId === model.id}
                />
              ))}
            </div>
          )}
        </div>

        {/* Right: predict tester */}
        <div className="col-span-2">
          <PredictTester productionModel={productionModel ?? null} />
        </div>
      </div>
    </div>
  )
}
