export default function ExperimentDashboard() {
  return (
    <div className="animate-slide-up space-y-6">
      <div>
        <h1 className="section-title">Experiment Dashboard</h1>
        <p className="text-sm text-slate-500 mt-1">Trigger multi-model training runs and track experiment history — Phase 1</p>
      </div>

      {/* Summary metric cards placeholder */}
      <div className="grid grid-cols-3 gap-4">
        {['Total Runs', 'Completed', 'In Progress'].map((label) => (
          <div key={label} className="card">
            <p className="metric-label">{label}</p>
            <p className="metric-value mt-2">—</p>
          </div>
        ))}
      </div>

      <div className="card">
        <p className="text-sm font-medium text-slate-400 mb-3">Experiment Runs</p>
        <div className="text-center py-10 text-slate-600 text-sm">
          No experiments yet — upload a dataset first
        </div>
      </div>
    </div>
  )
}
