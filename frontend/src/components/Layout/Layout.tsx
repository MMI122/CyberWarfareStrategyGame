import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Home, 
  Gamepad2, 
  Settings, 
  FlaskConical, 
  Shield,
  Zap 
} from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

const navItems = [
  { path: '/', icon: Home, label: 'Home' },
  { path: '/game', icon: Gamepad2, label: 'Play' },
  { path: '/research', icon: FlaskConical, label: 'Research' },
  { path: '/settings', icon: Settings, label: 'Settings' },
]

function Layout({ children }: LayoutProps) {
  const location = useLocation()
  
  return (
    <div className="min-h-screen bg-cyber-darker relative overflow-hidden">
      {/* Background grid effect */}
      <div className="fixed inset-0 bg-cyber-grid bg-[size:50px_50px] opacity-30 pointer-events-none" />
      
      {/* Gradient overlay */}
      <div className="fixed inset-0 bg-gradient-to-b from-cyber-dark/90 via-transparent to-cyber-dark/90 pointer-events-none" />
      
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 glass border-b border-cyber-primary/20">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <div className="relative">
                <Shield className="w-8 h-8 text-cyber-primary group-hover:text-cyber-secondary transition-colors" />
                <Zap className="w-4 h-4 text-cyber-accent absolute -bottom-1 -right-1" />
              </div>
              <div>
                <h1 className="font-mono font-bold text-lg text-cyber-primary group-hover:text-cyber-secondary transition-colors">
                  CYBER WARFARE
                </h1>
                <p className="text-xs text-gray-500 font-mono -mt-1">
                  STRATEGY GAME
                </p>
              </div>
            </Link>
            
            {/* Navigation */}
            <nav className="flex items-center gap-2">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path || 
                  (item.path === '/game' && location.pathname.startsWith('/game'))
                
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`
                      flex items-center gap-2 px-4 py-2 rounded-lg font-mono text-sm
                      transition-all duration-300
                      ${isActive 
                        ? 'bg-cyber-primary/20 text-cyber-primary border border-cyber-primary/50' 
                        : 'text-gray-400 hover:text-cyber-primary hover:bg-cyber-primary/10'}
                    `}
                  >
                    <item.icon className="w-4 h-4" />
                    <span className="hidden sm:inline">{item.label}</span>
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>
      </header>
      
      {/* Main content */}
      <main className="relative z-10 pt-16 min-h-screen">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          {children}
        </motion.div>
      </main>
      
      {/* Footer */}
      <footer className="relative z-10 border-t border-cyber-primary/20 glass">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-sm text-gray-500 font-mono">
            <p>© 2024 Cyber Warfare Strategy Game</p>
            <p className="flex items-center gap-2">
              <span className="w-2 h-2 bg-cyber-primary rounded-full animate-pulse" />
              AI Research Project
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default Layout
