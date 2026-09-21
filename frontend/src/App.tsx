import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'

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

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/datasets" replace />} />
            <Route
              path="datasets"
              element={
                <Suspense fallback={<PageLoader />}>
                  <DatasetUpload />
                </Suspense>
              }
            />
            <Route
              path="experiments"
              element={
                <Suspense fallback={<PageLoader />}>
                  <ExperimentDashboard />
                </Suspense>
              }
            />
            <Route
              path="models"
              element={
                <Suspense fallback={<PageLoader />}>
                  <ModelComparison />
                </Suspense>
              }
            />
            <Route
              path="observability"
              element={
                <Suspense fallback={<PageLoader />}>
                  <ObservabilityDashboard />
                </Suspense>
              }
            />
            <Route
              path="retrain"
              element={
                <Suspense fallback={<PageLoader />}>
                  <RetrainApproval />
                </Suspense>
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
