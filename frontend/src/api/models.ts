import { apiClient } from './client'

export type ModelStatus = 'staging' | 'production' | 'archived'
export type SlaTier = 'critical' | 'standard' | 'low'

export interface ModelVersion {
  id: string
  run_id: string
  registered_model_id: string | null
  registered_model_name: string | null
  version: number | null
  version_tag: string | null
  status: ModelStatus
  sla_tier: SlaTier
  deployed_at: string | null
  archived_at: string | null
  model_type: string | null
  task_type: string | null
  target_column: string | null
  metrics: Record<string, number | null> | null
  dataset_id: string | null
  input_columns: string[] | null
}

export interface ModelListResponse {
  items: ModelVersion[]
  total: number
}

export interface RegisteredModel {
  id: string
  name: string
  target_column: string
  task_type: string
  dataset_id: string | null
  created_at: string
  version_count: number
  production_version_id: string | null
  production_version_tag: string | null
  production_model_type: string | null
  production_sla_tier: SlaTier | null
  production_deployed_at: string | null
}

export interface PredictResponse {
  model_version_id: string
  prediction: unknown
  confidence: number | null
  probabilities: Record<string, number> | null
  prediction_id: string
  latency_ms: number
}

export const modelsApi = {
  list: async (params: { status?: ModelStatus; registeredModelId?: string } = {}): Promise<ModelListResponse> => {
    const res = await apiClient.get<ModelListResponse>('/models/', {
      params: { status_filter: params.status, registered_model_id: params.registeredModelId },
    })
    return res.data
  },

  registered: async (): Promise<RegisteredModel[]> => {
    const res = await apiClient.get<{ items: RegisteredModel[] }>('/models/registered')
    return res.data.items
  },

  get: async (id: string): Promise<ModelVersion> => {
    const res = await apiClient.get<ModelVersion>(`/models/${id}`)
    return res.data
  },

  deploy: async (id: string, slaTier: SlaTier = 'standard'): Promise<ModelVersion> => {
    const res = await apiClient.post<ModelVersion>(`/models/${id}/deploy`, { sla_tier: slaTier })
    return res.data
  },

  predict: async (id: string, features: Record<string, unknown>): Promise<PredictResponse> => {
    const res = await apiClient.post<PredictResponse>(`/models/${id}/predict`, { features })
    return res.data
  },

  sampleInput: async (id: string): Promise<Record<string, unknown>> => {
    const res = await apiClient.get<{ features: Record<string, unknown> }>(`/models/${id}/sample-input`)
    return res.data.features
  },
}
