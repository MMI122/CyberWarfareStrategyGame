// =============================================================================
// Game Types
// =============================================================================

// Type aliases for better compatibility
export type TopologyType = 'mesh' | 'star' | 'tree' | 'hybrid'
export type DifficultyLevel = 'easy' | 'medium' | 'hard' | 'expert'
export type AIType = 'minmax' | 'deep_rl'
export type PlayMode = 'vs_ai' | 'ai_vs_ai' | 'pvp'
export type Player = 'attacker' | 'defender'

export enum NodeType {
  WORKSTATION = 'WORKSTATION',
  SERVER = 'SERVER',
  ROUTER = 'ROUTER',
  FIREWALL = 'FIREWALL',
  DATABASE = 'DATABASE',
  DMZ = 'DMZ',
  HONEYPOT = 'HONEYPOT',
  ADMIN = 'ADMIN',
}

export enum NodeStatus {
  ONLINE = 'ONLINE',
  OFFLINE = 'OFFLINE',
  COMPROMISED = 'COMPROMISED',
  ISOLATED = 'ISOLATED',
  INFECTED = 'INFECTED',
}

export enum AccessLevel {
  NONE = 'NONE',
  USER = 'USER',
  ADMIN = 'ADMIN',
  ROOT = 'ROOT',
  SYSTEM = 'SYSTEM',
}

export enum PlayerRole {
  ATTACKER = 'ATTACKER',
  DEFENDER = 'DEFENDER',
}

export enum ActionType {
  SCAN = 'SCAN',
  EXPLOIT = 'EXPLOIT',
  PIVOT = 'PIVOT',
  EXFILTRATE = 'EXFILTRATE',
  INSTALL_BACKDOOR = 'INSTALL_BACKDOOR',
  CLEAN = 'CLEAN',
  ISOLATE = 'ISOLATE',
  DEPLOY_HONEYPOT = 'DEPLOY_HONEYPOT',
  PATCH = 'PATCH',
  RESTORE = 'RESTORE',
  MONITOR = 'MONITOR',
  END_TURN = 'END_TURN',
}

export enum GamePhase {
  SETUP = 'SETUP',
  PLAYING = 'PLAYING',
  PAUSED = 'PAUSED',
  GAME_OVER = 'GAME_OVER',
}

export interface Vulnerability {
  id: string
  name: string
  severity: number
  is_patched: boolean
  is_known: boolean
}

// Node type for UI (simplified)
export interface Node {
  id: string
  name: string
  type: string
  health: number
  criticality: number
  isCompromised: boolean
  isHoneypot: boolean
  controlledBy: Player | null
  vulnerabilities: Vulnerability[]
}

// Connection type for UI
export interface Connection {
  sourceId: string
  targetId: string
  isEncrypted: boolean
  bandwidth: number
}

// Network type for UI
export interface Network {
  nodes: Record<string, Node>
  connections: Connection[]
}

export interface NetworkNode {
  id: number
  name: string
  node_type: NodeType
  status: NodeStatus
  is_compromised: boolean
  is_visible: boolean
  access_level: AccessLevel
  defense_level: number
  importance: number
  vulnerabilities: Vulnerability[]
  connected_nodes: number[]
  position?: { x: number; y: number; z: number }
}

export interface NetworkEdge {
  source_id: number
  target_id: number
  is_active: boolean
  bandwidth: number
}

export interface GameAction {
  type: ActionType | string
  player: Player
  targetNodeId: string
  timestamp: string
}

export interface ActionResult {
  success: boolean
  message: string
  points_gained: number
}

// GameState for UI (matches backend response)
export interface GameState {
  game_id: string
  turn: number
  phase: GamePhase
  currentPlayer: Player
  scores: {
    attacker: number
    defender: number
  }
  network: Network
  valid_actions: GameAction[]
  gameOver: boolean
  winner: Player | null
  victory_condition: string | null
}

// =============================================================================
// API Types
// =============================================================================

export interface CreateGameRequest {
  difficulty: DifficultyLevel
  topology_type: TopologyType
  player_role: 'attacker' | 'defender'
  ai_type: AIType
}

export interface CreateGameResponse {
  game_id: string
  state: GameState
}

export interface ActionRequest {
  action_type: string
  target_node: number | null
  target_vulnerability?: string
}

export interface ActionResultResponse {
  success: boolean
  result: string
  state: GameState
}

export interface AIAgentInfo {
  name: string
  type: string
  description: string
  parameters: Record<string, string>
  strengths: string[]
  research_aspects: string[]
}

export interface MoveResponse {
  action: GameAction | null
  confidence: number
  evaluation: number
  reasoning: string
  time_taken: number
  nodes_searched: number | null
}

export interface ComparisonResult {
  winner: string
  final_scores: Record<string, number>
  total_turns: number
  moves: Array<{
    turn: number
    player: string
    agent: string
    action: string
    target: number | null
    success: boolean
  }>
  analysis: {
    games_played: number
    average_turns: number
    win_rate_agent1: number
    win_rate_agent2: number
  }
}

// =============================================================================
// WebSocket Types
// =============================================================================

export interface WSMessage {
  type: string
  [key: string]: unknown
}

export interface WSGameState extends WSMessage {
  type: 'game_state'
  state: GameState
}

export interface WSAction extends WSMessage {
  type: 'action_executed'
  player: string
  action: string
  target: number | null
  success: boolean
  message: string
}

export interface WSAIMove extends WSMessage {
  type: 'ai_move'
  action: GameAction | null
}

export interface WSError extends WSMessage {
  type: 'error'
  message: string
}

export interface WSGameOver extends WSMessage {
  type: 'game_over'
  winner: string
  condition: string
  message: string
}

// =============================================================================
// UI Types
// =============================================================================

export interface ToastNotification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  message: string
  duration?: number
}

export interface GameSettings {
  playMode: PlayMode
  topology: TopologyType
  difficulty: DifficultyLevel
  aiType: AIType
  aiDifficulty: DifficultyLevel
  aiTimeLimit: number
  soundEnabled: boolean
  showParticles: boolean
  animationSpeed: 'slow' | 'normal' | 'fast'
}
