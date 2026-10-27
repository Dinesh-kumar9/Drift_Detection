import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { Cpu, LogIn, UserPlus } from 'lucide-react'
import clsx from 'clsx'
import { errorMessage } from '../api/client'
import type { Role } from '../api/auth'
import { useAuth } from '../auth/AuthContext'

const DEMO_ACCOUNTS = [
  { email: 'admin@example.com', password: 'admin1234', role: 'admin' },
  { email: 'engineer@example.com', password: 'engineer1234', role: 'engineer' },
  { email: 'viewer@example.com', password: 'viewer1234', role: 'viewer' },
]

export default function Login() {
  const { user, login, register } = useAuth()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const next = params.get('next') || '/datasets'

  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<Role>('engineer')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to={next} replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      if (mode === 'login') await login(email, password)
      else await register(email, password, role)
      navigate(next, { replace: true })
    } catch (err) {
      setError(errorMessage(err, mode === 'login' ? 'Login failed' : 'Registration failed'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface px-4">
      <div className="w-full max-w-sm space-y-6 animate-slide-up">
        <div className="flex items-center gap-3 justify-center">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-accent flex items-center justify-center">
            <Cpu size={20} className="text-white" />
          </div>
          <div>
            <p className="text-lg font-bold text-slate-100 leading-tight">MLOps Platform</p>
            <p className="text-xs text-slate-500 font-mono">Observability</p>
          </div>
        </div>

        <form onSubmit={submit} className="card space-y-4">
          <div className="flex rounded-lg bg-surface p-1">
            {(['login', 'register'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => { setMode(m); setError(null) }}
                className={clsx(
                  'flex-1 py-1.5 text-xs font-medium rounded-md transition-colors',
                  mode === m ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-slate-200',
                )}
              >
                {m === 'login' ? 'Sign in' : 'Create account'}
              </button>
            ))}
          </div>

          <div>
            <label htmlFor="email" className="text-xs text-slate-500 mb-1.5 block">Email</label>
            <input id="email" type="email" className="input" value={email} autoComplete="email" required
              onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div>
            <label htmlFor="password" className="text-xs text-slate-500 mb-1.5 block">Password</label>
            <input id="password" type="password" className="input" value={password} required
              minLength={mode === 'register' ? 8 : undefined}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              onChange={(e) => setPassword(e.target.value)} />
          </div>
          {mode === 'register' && (
            <div>
              <label htmlFor="role" className="text-xs text-slate-500 mb-1.5 block">Role</label>
              <select id="role" className="input" value={role} onChange={(e) => setRole(e.target.value as Role)}>
                <option value="viewer">Viewer — read only</option>
                <option value="engineer">Engineer — train & deploy</option>
                <option value="admin">Admin — approvals</option>
              </select>
            </div>
          )}

          {error && (
            <p role="alert" className="text-xs text-drift-critical bg-drift-critical/10 border border-drift-critical/30 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <button type="submit" disabled={busy} className="btn-primary w-full justify-center">
            {busy
              ? <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              : mode === 'login' ? <LogIn size={14} /> : <UserPlus size={14} />}
            {mode === 'login' ? 'Sign in' : 'Create account'}
          </button>
        </form>

        {import.meta.env.DEV && mode === 'login' && (
          <div className="card py-3 space-y-1.5">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Local demo accounts</p>
            {DEMO_ACCOUNTS.map((a) => (
              <button
                key={a.email}
                type="button"
                onClick={() => { setEmail(a.email); setPassword(a.password) }}
                className="w-full flex justify-between text-xs font-mono text-slate-400 hover:text-slate-100 px-2 py-1 rounded hover:bg-surface-border/40"
              >
                <span>{a.email}</span>
                <span className="text-slate-600">{a.role}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
