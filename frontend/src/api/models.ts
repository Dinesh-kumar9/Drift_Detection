import { apiClient } from './client'

export interface ModelVersion {
  id: string
  run_id: string
  version_tag: string | null
  status: 'staging' | 'production' | 'archived'
  sla_tier: string
  deployed_at: string | null
  model_type: string | null
  metrics: Record<string, number> | null
  dataset_id: string | null
}

export interface ModelListResponse {
  items: ModelVersion[]
  total: number
}

export interface PredictResponse {
  model_version_id: string
  prediction: unknown
  confidence: number | null
  prediction_id: string
  latency_ms: number
}

export const modelsApi = {
  list: async (statusFilter?: string): Promise<ModelListResponse> => {
    const res = await apiClient.get<ModelListResponse>('/models/', {
      params: statusFilter ? { status_filter: statusFilter } : undefined,
    })
    return res.data
  },

  get: async (id: string): Promise<ModelVersion> => {
    const res = await apiClient.get<ModelVersion>(`/models/${id}`)
    return res.data
  },

  deploy: async (id: string, slaTier = 'standard'): Promise<ModelVersion> => {
    const res = await apiClient.post<ModelVersion>(`/models/${id}/deploy`, { sla_tier: slaTier })
    return res.data
  },

  predict: async (id: string, features: Record<string, unknown>): Promise<PredictResponse> => {
    const res = await apiClient.post<PredictResponse>(`/models/${id}/predict`, { features })
    return res.data
  },

  getProduction: async (): Promise<ModelVersion> => {
    const res = await apiClient.get<ModelVersion>('/models/production/current')
    return res.data
  },
}
