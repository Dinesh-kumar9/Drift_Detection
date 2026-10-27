import { lazy, Suspense, type ReactNode } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'
import AuthProvider from './auth/AuthProvider'
import RequireAuth from './auth/RequireAuth'
import Login from './pages/Login'

// ── Lazy-loaded pages ─────────────────────────────────────────────────────
const DatasetUpload        = lazy(() => import('./pages/DatasetUpload'))
const ExperimentDashboard  = lazy(() => import('./pages/ExperimentDashboard'))
const ModelComparison      = lazy(() => import('./pages/ModelComparison'))
const ObservabilityDashboard = lazy(() => import('./pages/ObservabilityDashboard'))
const RetrainApproval      = lazy(() => import('./pages/RetrainApproval'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
})

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

const page = (element: ReactNode) => <Suspense fallback={<PageLoader />}>{element}</Suspense>

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route
              path="/"
              element={
                <RequireAuth>
                  <Layout />
                </RequireAuth>
              }
            >
              <Route index element={<Navigate to="/datasets" replace />} />
              <Route path="datasets" element={page(<DatasetUpload />)} />
              <Route path="experiments" element={page(<ExperimentDashboard />)} />
              <Route path="models" element={page(<ModelComparison />)} />
              <Route path="observability" element={page(<ObservabilityDashboard />)} />
              <Route path="retrain" element={page(<RetrainApproval />)} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  )
}
