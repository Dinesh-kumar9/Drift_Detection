import { apiClient } from './client'

export type Role = 'viewer' | 'engineer' | 'admin'

export interface TokenResponse {
  access_token: string
  token_type: string
  role: Role
  email: string
}

export interface AuthUser {
  email: string
  role: Role
}

export const authApi = {
  login: async (email: string, password: string): Promise<TokenResponse> => {
    const res = await apiClient.post<TokenResponse>('/auth/login', { email, password })
    return res.data
  },

  register: async (email: string, password: string, role: Role = 'engineer'): Promise<TokenResponse> => {
    const res = await apiClient.post<TokenResponse>('/auth/register', { email, password, role })
    return res.data
  },
}
