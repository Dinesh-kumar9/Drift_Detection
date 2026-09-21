import { apiClient } from './client'

export interface MetricSet {
  model_type: string
  run_id: string
  accuracy?: number
  f1?: number
  precision?: number
  recall?: number
  roc_auc?: number
  mae?: number
  rmse?: number
  r2?: number
  latency_ms?: number
  training_time_s?: number
  status: string
}

export interface TrainResponse {
  experiment_group_id: string
  dataset_id: string
  target_column: string
  model_run_ids: string[]
  status: string
  message: string
}

export interface StatusResponse {
  experiment_group_id: string
  overall_status: string
  models: { model_type: string; run_id: string; status: string }[]
}

export interface ComparisonResponse {
  experiment_group_id: string
  dataset_id: string
  task_type: string
  models: MetricSet[]
  best_model_run_id: string | null
  best_metric: string | null
}

export interface ExperimentRun {
  id: string
  dataset_id: string
  model_type: string
  params: Record<string, unknown> | null
  metrics: Record<string, number> | null
  artifact_s3_path: string | null
  status: string
  created_at: string
  completed_at: string | null
}

export interface ExperimentListResponse {
  items: ExperimentRun[]
  total: number
}

export const experimentsApi = {
  train: async (payload: {
    dataset_id: string
    target_column: string
    task_type?: string
    test_size?: number
  }): Promise<TrainResponse> => {
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

  list: async (datasetId?: string, limit = 20): Promise<ExperimentListResponse> => {
    const res = await apiClient.get<ExperimentListResponse>('/experiments/', {
      params: { dataset_id: datasetId, limit },
    })
    return res.data
  },
}
