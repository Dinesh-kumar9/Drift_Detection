import { apiClient } from './client'

export type TaskType = 'auto' | 'classification' | 'regression'
export type RunState = 'queued' | 'running' | 'completed' | 'failed'

export interface MetricSet {
  model_type: string
  run_id: string
  model_version_id: string | null
  version_tag: string | null
  accuracy?: number | null
  f1?: number | null
  precision?: number | null
  recall?: number | null
  roc_auc?: number | null
  mae?: number | null
  rmse?: number | null
  r2?: number | null
  latency_ms?: number | null
  training_time_s?: number | null
  model_size_kb?: number | null
  status: RunState
  error: string | null
}

export interface TrainRequest {
  dataset_id: string
  target_column: string
  task_type?: TaskType
  models?: string[]
  test_size?: number
  model_name?: string
}

export interface TrainResponse {
  experiment_group_id: string
  dataset_id: string
  registered_model_id: string
  registered_model_name: string
  target_column: string
  task_type: string
  model_run_ids: string[]
  status: string
  message: string
}

export interface StatusResponse {
  experiment_group_id: string
  overall_status: RunState
  error: string | null
  models: { model_type: string; run_id: string; status: RunState; error: string | null }[]
}

export interface ComparisonResponse {
  experiment_group_id: string
  dataset_id: string
  registered_model_id: string | null
  task_type: string
  models: MetricSet[]
  best_model_run_id: string | null
  best_metric: string | null
}

export interface ExperimentGroup {
  id: string
  dataset_id: string
  dataset_name: string | null
  registered_model_id: string | null
  registered_model_name: string | null
  target_column: string
  task_type: string
  resolved_task_type: string | null
  status: RunState
  error: string | null
  n_models: number
  created_at: string
  completed_at: string | null
}

export interface ExperimentGroupListResponse {
  items: ExperimentGroup[]
  total: number
}

export const experimentsApi = {
  train: async (payload: TrainRequest): Promise<TrainResponse> => {
    const res = await apiClient.post<TrainResponse>('/experiments/train', payload)
    return res.data
  },

  status: async (groupId: string): Promise<StatusResponse> => {
    const res = await apiClient.get<StatusResponse>(`/experiments/${groupId}/status`)
    return res.data
  },

  compare: async (groupId: string): Promise<ComparisonResponse> => {
    const res = await apiClient.get<ComparisonResponse>(`/experiments/${groupId}/compare`)
    return res.data
  },

  groups: async (limit = 20): Promise<ExperimentGroupListResponse> => {
    const res = await apiClient.get<ExperimentGroupListResponse>('/experiments/groups', { params: { limit } })
    return res.data
  },
}
