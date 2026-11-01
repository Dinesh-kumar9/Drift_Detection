import { apiClient } from './client'
import type { Stage } from './observability'

export type RetrainStatus = 'pending' | 'approved' | 'running' | 'completed' | 'rejected' | 'failed'

export interface GateCheck {
  check: string
  passed: boolean
  detail: string
}

export interface RetrainJob {
  id: string
  model_version_id: string
  status: RetrainStatus
  trigger_type: 'auto' | 'manual' | null
  trigger_reason: { root_cause: Stage | null; data_since: string | null; note: string | null; forced: boolean } | null
  decision: {
    passed: boolean
    reason: string
    mode: 'post_onset_only' | 'original_plus_recent'
    labelled_rows: number
    baseline_error: number | null
    current_error: number | null
    error_increase: number | null
    cost_estimate: number | null
    expected_value: number | null
    checks: GateCheck[]
  } | null
  cost_estimate: number | null
  rejection_reason: string | null
  candidate_model_version_id: string | null
  evaluation: {
    holdout_rows: number
    metric: string
    production_error: number | null
    candidate_error: number | null
    improvement: number | null
    training_rows: number
    training_mode: string
    candidate_version_tag: string | null
  } | null
  error: string | null
  approved_at: string | null
  promoted_at: string | null
  created_at: string
  registered_model_name: string | null
  version_tag: string | null
  model_type: string | null
  sla_tier: string | null
}

export interface AuditLog {
  id: string
  actor_email: string | null
  action: string
  target_type: string | null
  target_id: string | null
  metadata: Record<string, unknown> | null
  created_at: string
}

export const retrainApi = {
  jobs: async (): Promise<RetrainJob[]> => {
    const res = await apiClient.get<{ items: RetrainJob[] }>('/retrain/jobs', { params: { limit: 100 } })
    return res.data.items
  },
  trigger: async (modelVersionId: string, note: string, force: boolean): Promise<RetrainJob> => {
    const res = await apiClient.post<RetrainJob>(`/retrain/${modelVersionId}/trigger`, { note: note || null, force })
    return res.data
  },
  approve: async (id: string): Promise<RetrainJob> => (await apiClient.post<RetrainJob>(`/retrain/jobs/${id}/approve`)).data,
  reject: async (id: string, reason: string): Promise<RetrainJob> =>
    (await apiClient.post<RetrainJob>(`/retrain/jobs/${id}/reject`, { reason })).data,
  promote: async (id: string): Promise<RetrainJob> => (await apiClient.post<RetrainJob>(`/retrain/jobs/${id}/promote`)).data,
  audit: async (limit = 50): Promise<AuditLog[]> => {
    const res = await apiClient.get<{ items: AuditLog[] }>('/audit-logs/', { params: { limit } })
    return res.data.items
  },
}
