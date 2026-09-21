import { apiClient } from './client'

export interface Dataset {
  id: string
  name: string
  s3_path: string
  row_count: string | null
  schema_json: Record<string, { dtype: string; null_count: number; null_pct: number; unique_count: number }> | null
  baseline_stats: Record<string, unknown> | null
  uploaded_at: string
  owner_id: string | null
}

export interface DatasetListResponse {
  items: Dataset[]
  total: number
  limit: number
  offset: number
}

export const datasetsApi = {
  upload: async (file: File, name: string): Promise<Dataset> => {
    const form = new FormData()
    form.append('file', file)
    form.append('name', name)
    const res = await apiClient.post<Dataset>('/datasets/', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return res.data
  },

  list: async (limit = 20, offset = 0): Promise<DatasetListResponse> => {
    const res = await apiClient.get<DatasetListResponse>('/datasets/', { params: { limit, offset } })
    return res.data
  },

  get: async (id: string): Promise<Dataset> => {
    const res = await apiClient.get<Dataset>(`/datasets/${id}`)
    return res.data
  },
}
