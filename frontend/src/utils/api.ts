import axios from 'axios'
import type {
  GameState,
  GameAction,
  CreateGameResponse,
  ActionResultResponse,
  MoveResponse,
  TopologyType,
  DifficultyLevel,
} from '../types'

// Configure axios with base URL
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// =============================================================================
// Game API
// =============================================================================

export const gameApi = {
  /**
   * Create a new game session
   */
  createGame: async (
    topology: TopologyType = 'mesh',
    difficulty: DifficultyLevel = 'medium'
  ): Promise<CreateGameResponse> => {
    const response = await api.post<CreateGameResponse>('/games/', {
      topology_type: topology,
      difficulty,
    })
    return response.data
  },
  
  /**
   * Get game state by ID
   */
  getGame: async (gameId: string): Promise<{ state: GameState }> => {
    const response = await api.get<{ state: GameState }>(`/games/${gameId}`)
    return response.data
  },
  
  /**
   * Execute a player action
   */
  performAction: async (gameId: string, action: GameAction): Promise<ActionResultResponse> => {
    const response = await api.post<ActionResultResponse>(`/games/${gameId}/action`, {
      action_type: action.type,
      target_node: action.targetNodeId,
    })
    return response.data
  },
  
  /**
   * Reset a game
   */
  resetGame: async (gameId: string): Promise<{ state: GameState }> => {
    const response = await api.post<{ state: GameState }>(`/games/${gameId}/reset`)
    return response.data
  },
  
  /**
   * Delete a game session
   */
  deleteGame: async (gameId: string): Promise<void> => {
    await api.delete(`/games/${gameId}`)
  },
}

// =============================================================================
// AI API
// =============================================================================

export const aiApi = {
  /**
   * Get MinMax move suggestion
   */
  getMinmaxMove: async (
    gameId: string,
    config?: { depth?: number; timeLimit?: number }
  ): Promise<MoveResponse> => {
    const response = await api.post<MoveResponse>(`/ai/minmax/${gameId}`, {
      depth: config?.depth ?? 4,
      time_limit: config?.timeLimit ?? 10000,
    })
    return response.data
  },
  
  /**
   * Get Deep RL move suggestion
   */
  getDeepRLMove: async (gameId: string): Promise<MoveResponse> => {
    const response = await api.post<MoveResponse>(`/ai/deeprl/${gameId}`)
    return response.data
  },
}

// =============================================================================
// WebSocket Connection
// =============================================================================

export class GameWebSocket {
  private ws: WebSocket | null = null
  private gameId: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  
  // Callbacks
  private stateUpdateCallback: ((state: GameState) => void) | null = null
  private aiMoveCallback: ((data: { action: GameAction | null }) => void) | null = null
  private errorCallback: ((error: { message: string }) => void) | null = null
  private gameOverCallback: ((data: { winner: string; condition: string }) => void) | null = null
  private connectCallback: (() => void) | null = null
  private disconnectCallback: (() => void) | null = null
  
  constructor(gameId: string) {
    this.gameId = gameId
  }
  
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/ws/game/${this.gameId}`
      
      this.ws = new WebSocket(wsUrl)
      
      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.reconnectAttempts = 0
        this.connectCallback?.()
        resolve()
      }
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.handleMessage(data)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        reject(error)
      }
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected')
        this.disconnectCallback?.()
        this.attemptReconnect()
      }
    })
  }
  
  private handleMessage(data: Record<string, unknown>) {
    switch (data.type) {
      case 'game_state':
        if (data.state) {
          this.stateUpdateCallback?.(data.state as GameState)
        }
        break
      case 'ai_move':
        this.aiMoveCallback?.({ action: (data.action as GameAction) || null })
        break
      case 'error':
        this.errorCallback?.({ message: String(data.message || 'Unknown error') })
        break
      case 'game_over':
        this.gameOverCallback?.({ 
          winner: String(data.winner || 'unknown'), 
          condition: String(data.condition || '') 
        })
        break
    }
  }
  
  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`Attempting reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`)
      setTimeout(() => this.connect(), 2000 * this.reconnectAttempts)
    }
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
  
  send(message: unknown) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    }
  }
  
  // Callback setters
  onStateUpdate(callback: (state: GameState) => void) {
    this.stateUpdateCallback = callback
  }
  
  onAIMove(callback: (data: { action: GameAction | null }) => void) {
    this.aiMoveCallback = callback
  }
  
  onError(callback: (error: { message: string }) => void) {
    this.errorCallback = callback
  }
  
  onGameOver(callback: (data: { winner: string; condition: string }) => void) {
    this.gameOverCallback = callback
  }
  
  onConnect(callback: () => void) {
    this.connectCallback = callback
  }
  
  onDisconnect(callback: () => void) {
    this.disconnectCallback = callback
  }
}

export default api
