import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ArrowLeft, 
  RotateCcw, 
  Bot, 
  User,
  Pause,
  Play,
  Settings2,
  Maximize2
} from 'lucide-react'
import { useGameStore, useSettingsStore, useAIStore } from '@/store'
import { gameApi, aiApi, GameWebSocket } from '@/utils/api'
import { NetworkGraph3D, ActionPanel, GameInfo } from '@/components/Game'
import { Button, Modal, useToast } from '@/components/UI'
import { GameAction, GameState } from '@/types'

function GamePage() {
  const navigate = useNavigate()
  const { addToast } = useToast()
  
  // Store state
  const { 
    gameId, 
    gameState, 
    setGameState, 
    setLoading, 
    isLoading,
    error,
    setError 
  } = useGameStore()
  const { settings } = useSettingsStore()
  const { setThinking, isThinking } = useAIStore()
  
  // Local state
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [, setWs] = useState<GameWebSocket | null>(null)
  const [isPaused, setIsPaused] = useState(false)
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [, setIsFullscreen] = useState(false)
  
  // Load initial game state
  useEffect(() => {
    const loadGameState = async () => {
      if (!gameId) return
      
      setLoading(true)
      try {
        const gameState = await gameApi.getGame(gameId)
        if (gameState) {
          setGameState(gameState)
        }
      } catch (err) {
        console.error('Failed to load game:', err)
        setError('Failed to load game state')
        addToast('error', 'Failed to load game. Returning to home.')
        navigate('/')
      } finally {
        setLoading(false)
      }
    }
    
    loadGameState()
  }, [gameId])
  
  // Initialize WebSocket connection (optional, for real-time updates)
  useEffect(() => {
    if (gameId && gameState) {
      const websocket = new GameWebSocket(gameId)
      
      websocket.onStateUpdate((state) => {
        setGameState(state)
      })
      
      websocket.onAIMove((data) => {
        addToast('info', `AI played: ${data.action?.type || 'move'}`)
        setThinking(false)
      })
      
      websocket.onError((error) => {
        addToast('error', error.message)
      })
      
      websocket.onGameOver((data) => {
        addToast(
          data.winner === 'defender' ? 'success' : 'warning',
          `Game Over! ${data.winner.charAt(0).toUpperCase() + data.winner.slice(1)} wins!`
        )
      })
      
      websocket.connect().catch(err => {
        console.warn('WebSocket connection failed, game will work without real-time updates:', err)
      })
      setWs(websocket)
      
      return () => {
        websocket.disconnect()
      }
    }
  }, [gameId, gameState !== null])
  
  // Redirect if no game
  useEffect(() => {
    if (!gameId && !isLoading) {
      navigate('/')
    }
  }, [gameId, isLoading, navigate])
  
  // Handle player action
  const handleAction = useCallback(async (action: GameAction) => {
    if (!gameId || !gameState) return
    
    setLoading(true)
    setError(null)
    
    try {
      const result = await gameApi.performAction(gameId, action)
      setGameState(result.game_state)
      
      if (result.message) {
        addToast('success', `Action successful: ${result.message}`)
      }
      
      // Check if game over
      if (result.game_state.game_over) {
        addToast(
          result.game_state.winner === 'DEFENDER' ? 'success' : 'warning',
          `Game Over! ${result.game_state.winner} wins!`
        )
        return
      }
      
      // If it's now AI's turn, trigger AI move
      if (settings.playMode === 'vs_ai' && result.game_state.current_player !== 'DEFENDER') {
        await triggerAIMove(result.game_state)
      }
    } catch (err) {
      console.error('Action failed:', err)
      addToast('error', 'Action failed. Try again.')
      setError('Action failed')
    } finally {
      setLoading(false)
    }
  }, [gameId, gameState, settings.playMode])
  
  // Trigger AI move
  const triggerAIMove = useCallback(async (_currentState: GameState) => {
    if (!gameId) return
    
    setThinking(true)
    
    try {
      let result
      
      if (settings.aiType === 'minmax') {
        result = await aiApi.getMinmaxMove(gameId, {
          depth: settings.aiDifficulty === 'easy' ? 2 : 
                 settings.aiDifficulty === 'medium' ? 4 :
                 settings.aiDifficulty === 'hard' ? 6 : 8,
          timeLimit: 10000
        })
      } else {
        result = await aiApi.getDeepRLMove(gameId)
      }
      
      if (result.action) {
        // Execute the AI's chosen action
        const actionResult = await gameApi.performAction(gameId, result.action)
        setGameState(actionResult.game_state)
        
        addToast('info', `AI played: ${result.action.type}`)
      }
    } catch (err) {
      console.error('AI move failed:', err)
      addToast('error', 'AI failed to make a move')
    } finally {
      setThinking(false)
    }
  }, [gameId, settings.aiType, settings.aiDifficulty])
  
  // Handle reset - navigate to home to start new game
  const handleReset = useCallback(async () => {
    navigate('/')
    addToast('info', 'Start a new game from the home page')
  }, [navigate])
  
  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }
  
  // Determine if it's player's turn
  const isPlayerTurn = gameState?.current_player === 'DEFENDER' || settings.playMode === 'pvp'
  
  if (!gameState) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-cyber-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400 font-mono">Loading game...</p>
        </div>
      </div>
    )
  }
  
  return (
    <div className="container mx-auto px-4 py-6">
      {/* Top Bar */}
      <motion.div 
        className="flex items-center justify-between mb-6"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/')}
            leftIcon={<ArrowLeft className="w-4 h-4" />}
          >
            Back
          </Button>
          
          <div className="text-gray-400 font-mono text-sm">
            Game ID: <span className="text-cyber-primary">{gameId?.slice(0, 8)}...</span>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {/* AI indicator */}
          <AnimatePresence>
            {isThinking && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyber-secondary/20 border border-cyber-secondary/50"
              >
                <Bot className="w-4 h-4 text-cyber-secondary animate-pulse" />
                <span className="text-cyber-secondary text-sm font-mono">AI Thinking...</span>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Current player indicator */}
          <div className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg border
            ${gameState.currentPlayer === 'attacker' 
              ? 'bg-red-500/10 border-red-500/50 text-red-400' 
              : 'bg-green-500/10 border-green-500/50 text-green-400'}
          `}>
            {gameState.currentPlayer === 'attacker' ? (
              <Bot className="w-4 h-4" />
            ) : (
              <User className="w-4 h-4" />
            )}
            <span className="text-sm font-mono capitalize">{gameState.currentPlayer}'s Turn</span>
          </div>
          
          {/* Actions */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsPaused(!isPaused)}
            leftIcon={isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
          >
            {isPaused ? 'Resume' : 'Pause'}
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            leftIcon={<RotateCcw className="w-4 h-4" />}
          >
            Reset
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSettingsModal(true)}
            leftIcon={<Settings2 className="w-4 h-4" />}
          />
          
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleFullscreen}
            leftIcon={<Maximize2 className="w-4 h-4" />}
          />
        </div>
      </motion.div>
      
      {/* Main Game Area */}
      <div className="grid lg:grid-cols-4 gap-6">
        {/* Network Graph (3D Visualization) */}
        <div className="lg:col-span-3 h-[600px]">
          <NetworkGraph3D
            gameState={gameState}
            onNodeClick={setSelectedNode}
            selectedNode={selectedNode}
          />
        </div>
        
        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Game Info */}
          <GameInfo gameState={gameState} />
          
          {/* Action Panel */}
          <ActionPanel
            gameState={gameState}
            selectedNode={selectedNode}
            onActionSelect={handleAction}
            isPlayerTurn={isPlayerTurn && !isPaused}
            isProcessing={isLoading || isThinking}
          />
        </div>
      </div>
      
      {/* Settings Modal */}
      <Modal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
        title="Game Settings"
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">AI Type</label>
            <div className="text-white font-mono">{settings.aiType}</div>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Difficulty</label>
            <div className="text-white font-mono capitalize">{settings.difficulty}</div>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Network Topology</label>
            <div className="text-white font-mono capitalize">{settings.topology}</div>
          </div>
        </div>
      </Modal>
      
      {/* Error display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-4 left-4 p-4 rounded-lg bg-red-500/20 border border-red-500 text-red-400"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default GamePage
