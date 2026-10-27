import { useCallback, useMemo, useState, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { authApi, type AuthUser, type Role, type TokenResponse } from '../api/auth'
import { TOKEN_KEY, USER_KEY } from '../api/client'
import { AuthContext, type AuthContextValue } from './AuthContext'

function readStoredUser(): AuthUser | null {
  if (!localStorage.getItem(TOKEN_KEY)) return null
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) ?? 'null')
  } catch {
    return null
  }
}

export default function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  const [user, setUser] = useState<AuthUser | null>(readStoredUser)

  const storeSession = useCallback((res: TokenResponse) => {
    const next = { email: res.email, role: res.role }
    localStorage.setItem(TOKEN_KEY, res.access_token)
    localStorage.setItem(USER_KEY, JSON.stringify(next))
    setUser(next)
  }, [])

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      login: async (email, password) => storeSession(await authApi.login(email, password)),
      register: async (email, password, role?: Role) => storeSession(await authApi.register(email, password, role)),
      logout: () => {
        localStorage.removeItem(TOKEN_KEY)
        localStorage.removeItem(USER_KEY)
        queryClient.clear() // don't leak the previous user's cached data
        setUser(null)
      },
    }),
    [user, storeSession, queryClient],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
