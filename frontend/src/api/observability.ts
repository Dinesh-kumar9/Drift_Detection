import { apiClient } from './client'

export type Stage = 'S1_ingestion' | 'S2_preprocessing' | 'S3_output'
export const STAGES: Stage[] = ['S1_ingestion', 'S2_preprocessing', 'S3_output']
export type FaultType = 'none' | 's1_shift' | 's1_nulls' | 's2_shift' | 's3_concept'

export interface FeatureTest {
  feature: string
  test: string
  statistic: number
  p_value: number
  drifted: boolean
  n_reference: number
  n_current: number
  detail: Record<string, unknown>
}

export interface StageStatus {
  stage: Stage
  score: number
  drifted: boolean
  p_value: number | null
  detector: string | null
  top_feature: string | null
  onset: string | null
  n_samples: number | null
  details: { tests?: FeatureTest[]; signals?: Record<string, unknown> } | null
}

export interface DriftStatus {
  model_version_id: string
  threshold: number
  latest: {
    check_id: string | null
    detected_at: string
    window_start: string | null
    window_end: string | null
    stages: StageStatus[]
  } | null
  series: {
    detected_at: string
    check_id: string | null
    scores: Partial<Record<Stage, number>>
    drifted: Partial<Record<Stage, boolean>>
  }[]
}

export interface RankedStage {
  stage: Stage
  score: number
  confidence: number
  magnitude: number
  temporal_weight: number
  causal_factor: number
  onset: string | null
  onset_lag_seconds: number
  top_feature: string | null
  reason: string
}

export interface AttributionReport {
  id: string
  model_version_id: string | null
  check_id: string | null
  root_cause_stage: Stage | null
  confidence: number | null
  explanation: string | null
  ranked_stages: RankedStage[]
  window_start: string | null
  window_end: string | null
  alert_sent_at: string | null
  generated_at: string
}

export interface OverviewItem {
  model_version_id: string
  registered_model_name: string | null
  version_tag: string | null
  model_type: string | null
  sla_tier: string
  deployed_at: string | null
  status: 'healthy' | 'drifting' | 'not_checked' | 'no_baseline'
  last_check_at: string | null
  stages: Partial<Record<Stage, { score: number; drifted: boolean }>>
  root_cause: Stage | null
  confidence: number | null
  predictions_24h: number
}

export interface CheckOutcome {
  status: 'completed' | 'insufficient_data' | 'no_baseline'
  drift_detected?: boolean
  root_cause?: Stage | null
  confidence?: number | null
  explanation?: string
  n_samples?: number
  message?: string
  alert?: string | null
}

export interface SimulateRequest {
  fault: FaultType
  feature?: string
  magnitude?: number
  n?: number
  fault_fraction?: number
  span_minutes?: number
}

export const observabilityApi = {
  overview: async (): Promise<OverviewItem[]> => {
    const res = await apiClient.get<{ items: OverviewItem[] }>('/observability/overview')
    return res.data.items
  },

  drift: async (id: string, hours = 24): Promise<DriftStatus> => {
    const res = await apiClient.get<DriftStatus>(`/observability/${id}/drift`, { params: { hours } })
    return res.data
  },

  attribution: async (id: string): Promise<{ latest: AttributionReport | null; history: AttributionReport[] }> => {
    const res = await apiClient.get(`/observability/${id}/attribution`)
    return res.data
  },

  check: async (id: string): Promise<CheckOutcome> => {
    const res = await apiClient.post<CheckOutcome>(`/observability/${id}/check`)
    return res.data
  },

  simulate: async (id: string, req: SimulateRequest): Promise<{ traffic: Record<string, unknown>; check: CheckOutcome | null }> => {
    const res = await apiClient.post(`/observability/${id}/simulate`, req, { timeout: 120_000 })
    return res.data
  },
}
