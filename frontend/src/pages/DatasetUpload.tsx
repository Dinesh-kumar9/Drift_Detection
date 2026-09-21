// ── Stub pages — full implementation in Phase 1 ───────────────────────────

export default function DatasetUpload() {
  return (
    <div className="animate-slide-up space-y-6">
      <div>
        <h1 className="section-title">Dataset Upload</h1>
        <p className="text-sm text-slate-500 mt-1">Upload CSV datasets with schema validation — Phase 1</p>
      </div>

      {/* Upload zone placeholder */}
      <div className="card-glass border-2 border-dashed border-surface-border hover:border-brand-500/50 transition-colors duration-300 flex flex-col items-center justify-center py-20 cursor-pointer group">
        <div className="w-16 h-16 rounded-2xl bg-brand-600/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
          <svg className="w-8 h-8 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
        </div>
        <p className="text-slate-300 font-medium">Drop your CSV file here</p>
        <p className="text-slate-600 text-sm mt-1">or <span className="text-brand-400 hover:underline">browse files</span></p>
        <p className="text-slate-700 text-xs mt-4 font-mono">Implementation: Phase 1</p>
      </div>

      {/* Empty state datasets list */}
      <div className="card">
        <p className="text-sm font-medium text-slate-400 mb-3">Recent Datasets</p>
        <div className="text-center py-10 text-slate-600 text-sm">No datasets uploaded yet</div>
      </div>
    </div>
  )
}
