export default function ObservabilityDashboard() {
  return (
    <div className="animate-slide-up space-y-6">
      <div>
        <h1 className="section-title">Observability Dashboard</h1>
        <p className="text-sm text-slate-500 mt-1">
          Stage-wise drift detection (S1 Ingestion → S2 Preprocessing → S3 Predictions) with root-cause attribution — Phase 2
        </p>
      </div>

      {/* Stage cards */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { stage: 'S1 — Ingestion',      color: 'from-brand-600/20 to-brand-800/10' },
          { stage: 'S2 — Preprocessing',  color: 'from-accent/20 to-accent/5' },
          { stage: 'S3 — Predictions',    color: 'from-amber-600/20 to-amber-800/10' },
        ].map(({ stage, color }) => (
          <div key={stage} className={`card bg-gradient-to-br ${color}`}>
            <p className="text-xs font-mono text-slate-500 uppercase tracking-widest">{stage}</p>
            <p className="metric-value mt-2">—</p>
            <p className="metric-label mt-1">Drift Score</p>
            <div className="mt-3 h-1 rounded-full bg-surface-border">
              <div className="h-1 rounded-full bg-surface-muted w-0" />
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <p className="text-sm font-medium text-slate-400 mb-3">Root Cause Attribution</p>
        <div className="text-center py-10 text-slate-600 text-sm">
          Deploy a model to begin monitoring
        </div>
      </div>
    </div>
  )
}
