import { Outlet, NavLink, useLocation } from 'react-router-dom'
import {
  Database, FlaskConical, GitCompare, Activity, RefreshCcw,
  Cpu, ChevronRight, Bell,
} from 'lucide-react'
import clsx from 'clsx'

const NAV_ITEMS = [
  { to: '/datasets',      label: 'Datasets',       icon: Database,      phase: 1 },
  { to: '/experiments',   label: 'Experiments',     icon: FlaskConical,  phase: 1 },
  { to: '/models',        label: 'Model Comparison',icon: GitCompare,    phase: 1 },
  { to: '/observability', label: 'Observability',   icon: Activity,      phase: 2 },
  { to: '/retrain',       label: 'Retrain Console', icon: RefreshCcw,    phase: 3 },
]

export default function Layout() {
  const location = useLocation()

  return (
    <div className="flex h-screen overflow-hidden bg-surface">
      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="w-64 flex-shrink-0 flex flex-col border-r border-surface-border bg-surface-card/50 backdrop-blur-sm">
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-surface-border">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-accent flex items-center justify-center">
            <Cpu size={16} className="text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-slate-100 leading-tight">MLOps Platform</p>
            <p className="text-[10px] text-slate-500 font-mono">Observability v0.1</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <p className="px-3 mb-2 text-[10px] font-semibold text-slate-600 uppercase tracking-widest">
            Platform
          </p>
          {NAV_ITEMS.map(({ to, label, icon: Icon, phase }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                clsx(isActive ? 'nav-link-active' : 'nav-link')
              }
            >
              <Icon size={16} className="flex-shrink-0" />
              <span className="flex-1">{label}</span>
              <span className={clsx(
                'text-[9px] font-mono px-1.5 py-0.5 rounded',
                phase === 1 && 'bg-brand-900/50 text-brand-400',
                phase === 2 && 'bg-accent/10 text-accent',
                phase === 3 && 'bg-amber-900/40 text-amber-400',
              )}>
                P{phase}
              </span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-4 border-t border-surface-border">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-brand-400 to-accent flex items-center justify-center text-[11px] font-bold text-white">
              U
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-slate-300 truncate">MLOps Engineer</p>
              <p className="text-[10px] text-slate-600">engineer</p>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main content ──────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-surface-border bg-surface-card/30 backdrop-blur-sm flex-shrink-0">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span>Platform</span>
            <ChevronRight size={14} />
            <span className="text-slate-200 capitalize font-medium">
              {location.pathname.split('/')[1] || 'Overview'}
            </span>
          </div>

          {/* Header actions */}
          <div className="flex items-center gap-3">
            {/* Live indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface border border-surface-border text-xs text-slate-400">
              <span className="w-1.5 h-1.5 rounded-full bg-drift-safe animate-pulse-dot" />
              All systems operational
            </div>
            <button className="relative p-2 rounded-lg hover:bg-surface-border text-slate-400 hover:text-slate-200 transition-colors">
              <Bell size={16} />
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-accent" />
            </button>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
