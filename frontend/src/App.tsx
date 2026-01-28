import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { Layout } from '@/components/Layout'
import { ToastProvider } from '@/components/UI'
import { HomePage, GamePage, SettingsPage, ResearchPage } from '@/pages'

function App() {
  return (
    <Router>
      <ToastProvider>
        <Layout>
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/game/:gameId?" element={<GamePage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/research" element={<ResearchPage />} />
            </Routes>
          </AnimatePresence>
        </Layout>
      </ToastProvider>
    </Router>
  )
}

export default App
