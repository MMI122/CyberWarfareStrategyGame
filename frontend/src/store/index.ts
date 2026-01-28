import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { 
  GameState, 
  GameAction, 
  GameSettings
} from '../types'

// =============================================================================
// Game Store
// =============================================================================

interface GameStore {
  // Game state
  gameId: string | null
  gameState: GameState | null
  isLoading: boolean
  error: string | null
  
  // Actions
  setGameId: (id: string | null) => void
  setGameState: (state: GameState | null) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  resetGame: () => void
}

export const useGameStore = create<GameStore>((set) => ({
  gameId: null,
  gameState: null,
  isLoading: false,
  error: null,
  
  setGameId: (gameId) => set({ gameId }),
  setGameState: (gameState) => set({ gameState, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),
  resetGame: () => set({
    gameId: null,
    gameState: null,
    error: null,
  }),
}))

// =============================================================================
// Settings Store
// =============================================================================

interface SettingsStore {
  settings: GameSettings
  updateSettings: (settings: Partial<GameSettings>) => void
  resetSettings: () => void
}

const defaultSettings: GameSettings = {
  playMode: 'vs_ai',
  topology: 'mesh',
  difficulty: 'medium',
  aiType: 'minmax',
  aiDifficulty: 'medium',
  aiTimeLimit: 10000,
  soundEnabled: true,
  showParticles: true,
  animationSpeed: 'normal',
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      settings: defaultSettings,
      
      updateSettings: (newSettings) => set((state) => ({
        settings: { ...state.settings, ...newSettings }
      })),
      
      resetSettings: () => set({ settings: defaultSettings }),
    }),
    {
      name: 'cyber-warfare-settings',
    }
  )
)

// =============================================================================
// AI Store
// =============================================================================

interface AIStore {
  isThinking: boolean
  lastMove: {
    action: GameAction | null
    confidence: number
    reasoning: string
  } | null
  
  setThinking: (thinking: boolean) => void
  setLastMove: (move: AIStore['lastMove']) => void
}

export const useAIStore = create<AIStore>((set) => ({
  isThinking: false,
  lastMove: null,
  
  setThinking: (isThinking) => set({ isThinking }),
  setLastMove: (lastMove) => set({ lastMove }),
}))
