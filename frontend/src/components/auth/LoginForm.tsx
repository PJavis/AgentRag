'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/lib/stores/auth-store'
import { getApiUrl, getConfig } from '@/lib/config'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { AlertCircle } from 'lucide-react'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

type Mode = 'login' | 'signup'

interface AuthStatus {
  auth_enabled: boolean
  allow_signup?: boolean
  providers?: { password?: boolean; google?: boolean }
}

export function LoginForm() {
  const router = useRouter()
  const { authRequired, checkAuthRequired, isAuthenticated, hasHydrated } = useAuthStore()
  const setLogin = useAuthStore((s) => s.login) // legacy bearer login
  const setAuthState = (token: string) => {
    useAuthStore.setState({
      isAuthenticated: true,
      token,
      lastAuthCheck: Date.now(),
      isLoading: false,
      error: null,
    })
  }

  const [status, setStatus] = useState<AuthStatus | null>(null)
  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isCheckingAuth, setIsCheckingAuth] = useState(true)
  const [configInfo, setConfigInfo] = useState<{ apiUrl: string; version: string } | null>(null)

  useEffect(() => {
    void getConfig().then((cfg) =>
      setConfigInfo({ apiUrl: cfg.apiUrl, version: cfg.version })
    ).catch(() => {})
  }, [])

  // ── First, consume token from URL fragment after Google redirect.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const hash = window.location.hash || ''
    const m = hash.match(/[#&]token=([^&]+)/)
    if (m && m[1]) {
      const token = decodeURIComponent(m[1])
      setAuthState(token)
      window.history.replaceState({}, '', window.location.pathname)
      router.push('/notebooks')
    }
  }, [router])

  // ── Discover whether auth is required + which providers are available.
  useEffect(() => {
    if (!hasHydrated) return
    let cancelled = false
    const run = async () => {
      try {
        const apiUrl = await getApiUrl()
        const resp = await fetch(`${apiUrl}/api/auth/status`, { cache: 'no-store' })
        if (resp.ok && !cancelled) {
          const data: AuthStatus = await resp.json()
          setStatus(data)
          if (!data.auth_enabled) {
            setAuthState('not-required')
            router.push('/notebooks')
            return
          }
        }
        await checkAuthRequired().catch(() => {})
      } catch (err) {
        console.error(err)
      } finally {
        if (!cancelled) setIsCheckingAuth(false)
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [hasHydrated, checkAuthRequired, router])

  useEffect(() => {
    if (isAuthenticated && hasHydrated && !isCheckingAuth) {
      router.push('/notebooks')
    }
  }, [isAuthenticated, hasHydrated, isCheckingAuth, router])

  if (!hasHydrated || isCheckingAuth) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <LoadingSpinner />
      </div>
    )
  }

  const allowSignup = status?.allow_signup !== false
  const googleEnabled = !!status?.providers?.google
  const showLegacyOnly = !status // server didn't respond → fallback

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    if (!email.trim() || !password.trim()) return
    setIsSubmitting(true)
    try {
      const apiUrl = await getApiUrl()
      const path = mode === 'signup' ? '/api/auth/signup' : '/api/auth/login'
      const body: Record<string, string> = { email: email.trim(), password }
      if (mode === 'signup' && name.trim()) body.name = name.trim()
      const resp = await fetch(`${apiUrl}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}))
        setError(detail.detail || `Authentication failed (${resp.status})`)
        setIsSubmitting(false)
        return
      }
      const data = await resp.json()
      const token: string | undefined = data?.access_token
      if (!token) {
        setError('Server did not return a token')
        setIsSubmitting(false)
        return
      }
      setAuthState(token)
      router.push('/notebooks')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error')
      setIsSubmitting(false)
    }
  }

  const handleLegacySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!password.trim()) return
    setError(null)
    await setLogin(password)
  }

  const handleGoogle = async () => {
    const apiUrl = await getApiUrl()
    window.location.href = `${apiUrl}/api/auth/google/start`
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle>{mode === 'signup' ? 'Tạo tài khoản' : 'Đăng nhập AgentRag'}</CardTitle>
          <CardDescription>
            {mode === 'signup'
              ? 'Đăng ký để truy cập notebook học liệu của bạn.'
              : 'Đăng nhập bằng email & mật khẩu hoặc Google.'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {showLegacyOnly ? (
            <form onSubmit={handleLegacySubmit} className="space-y-4">
              <Input
                type="password"
                placeholder="Mật khẩu (legacy)"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <Button type="submit" className="w-full" disabled={!password.trim()}>
                Đăng nhập
              </Button>
            </form>
          ) : (
            <>
              <div className="flex border rounded-md overflow-hidden mb-4">
                <button
                  type="button"
                  className={`flex-1 py-2 text-sm transition ${
                    mode === 'login'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-background hover:bg-accent'
                  }`}
                  onClick={() => {
                    setMode('login')
                    setError(null)
                  }}
                >
                  Đăng nhập
                </button>
                {allowSignup && (
                  <button
                    type="button"
                    className={`flex-1 py-2 text-sm transition ${
                      mode === 'signup'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-background hover:bg-accent'
                    }`}
                    onClick={() => {
                      setMode('signup')
                      setError(null)
                    }}
                  >
                    Đăng ký
                  </button>
                )}
              </div>

              <form onSubmit={handleEmailSubmit} className="space-y-3">
                {mode === 'signup' && (
                  <Input
                    type="text"
                    placeholder="Tên hiển thị (tuỳ chọn)"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    disabled={isSubmitting}
                  />
                )}
                <Input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={isSubmitting}
                  autoComplete="email"
                  required
                />
                <Input
                  type="password"
                  placeholder={mode === 'signup' ? 'Mật khẩu (≥ 6 ký tự)' : 'Mật khẩu'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isSubmitting}
                  autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                  required
                  minLength={6}
                />
                {error && (
                  <div className="flex items-center gap-2 text-red-600 text-sm">
                    <AlertCircle className="h-4 w-4 shrink-0" />
                    <span>{error}</span>
                  </div>
                )}
                <Button
                  type="submit"
                  className="w-full"
                  disabled={isSubmitting || !email.trim() || !password.trim()}
                >
                  {isSubmitting
                    ? mode === 'signup'
                      ? 'Đang tạo...'
                      : 'Đang đăng nhập...'
                    : mode === 'signup'
                      ? 'Tạo tài khoản'
                      : 'Đăng nhập'}
                </Button>
              </form>

              {googleEnabled && (
                <>
                  <div className="my-4 flex items-center gap-2 text-xs text-muted-foreground">
                    <div className="flex-1 border-t" />
                    <span>hoặc</span>
                    <div className="flex-1 border-t" />
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full"
                    onClick={handleGoogle}
                  >
                    <span className="mr-2">G</span>
                    Tiếp tục với Google
                  </Button>
                </>
              )}
            </>
          )}

          {configInfo && (
            <div className="mt-4 text-[10px] text-center text-muted-foreground border-t pt-2">
              <div>v{configInfo.version}</div>
              <div className="font-mono break-all">{configInfo.apiUrl}</div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
