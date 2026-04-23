import { ReactNode } from 'react'
import { Link } from 'react-router-dom'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--kv-n-50)', color: 'var(--kv-ink)' }}>
      {/* NAV */}
      <nav style={{
        height: 64, background: 'var(--kv-white)',
        borderBottom: '1px solid var(--kv-n-200)',
        display: 'grid', gridTemplateColumns: 'auto 1fr auto',
        alignItems: 'center', padding: '0 28px', gap: 24,
        position: 'sticky', top: 0, zIndex: 20,
      }}>
        <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 14, textDecoration: 'none' }}>
          <span style={{
            fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 17,
            color: 'var(--kv-black)', textTransform: 'lowercase' as const, letterSpacing: '-0.01em'
          }}>
            kinetic vision<span className="kv-dot">.</span>
          </span>
          <span style={{ width: 1, height: 28, background: 'var(--kv-n-200)' }} />
          <span style={{ display: 'flex', flexDirection: 'column' as const, lineHeight: 1.1 }}>
            <span className="kv-eyebrow">Industrial Engineering</span>
            <span style={{
              fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 15,
              color: 'var(--kv-black)', marginTop: 2, letterSpacing: '-0.005em'
            }}>
              Real Time Vision<span className="kv-dot">.</span>
            </span>
          </span>
        </Link>

        <div style={{
          justifySelf: 'center', display: 'inline-flex', alignItems: 'center', gap: 10,
          padding: '6px 12px 6px 10px', border: '1px solid var(--kv-n-200)',
          borderRadius: 9999, background: 'var(--kv-white)', fontSize: 12, color: 'var(--kv-n-600)',
        }}>
          <span className="dot-live" style={{
            width: 8, height: 8, borderRadius: '50%', background: 'var(--sem-success)',
          }} />
          <span>Engagement · <b style={{ color: 'var(--kv-ink)' }}>Ashley Furniture</b> · Warehouse Vision Pilot</span>
        </div>

        <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
          <Link to="/" style={{
            fontSize: 13, fontWeight: 700, letterSpacing: '0.02em', color: 'var(--kv-ink)',
          }}>Upload</Link>
          <a href="#" style={{
            fontSize: 13, fontWeight: 700, letterSpacing: '0.02em', color: 'var(--kv-n-500)',
          }}>Dashboard</a>
        </div>
      </nav>

      <main style={{ maxWidth: 1280, margin: '0 auto', padding: '32px 28px' }}>
        {children}
      </main>

      {/* FOOTER */}
      <footer style={{
        marginTop: 48, padding: '20px 28px', borderTop: '1px solid var(--kv-n-200)',
        background: 'var(--kv-white)', fontFamily: 'var(--font-mono)', fontSize: 11,
        color: 'var(--kv-n-500)', letterSpacing: '0.04em',
        display: 'flex', justifyContent: 'space-between',
      }}>
        <span>KINETIC VISION · REAL TIME VISION</span>
        <span>CONFIDENTIAL · ASHLEY FURNITURE PILOT · v0.2</span>
      </footer>
    </div>
  )
}
