import { motion } from 'framer-motion'
import { 
  Shield, 
  Sword, 
  Clock, 
  Target, 
  Activity,
  Trophy,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react'
import { GameState, Player } from '@/types'

interface GameInfoProps {
  gameState: GameState
}

function PlayerScore({ player, score, isActive, stats }: { 
  player: Player
  score: number
  isActive: boolean
  stats: { compromised: number; secured: number }
}) {
  const isAttacker = player === 'attacker'
  const Icon = isAttacker ? Sword : Shield
  const color = isAttacker ? 'red-400' : 'green-400'
  const bgGradient = isAttacker 
    ? 'from-red-500/20 to-orange-500/20' 
    : 'from-green-500/20 to-cyan-500/20'
  
  return (
    <motion.div 
      className={`
        flex-1 p-4 rounded-xl border-2 transition-all
        ${isActive 
          ? `border-${color} bg-gradient-to-br ${bgGradient}` 
          : 'border-gray-700 bg-cyber-dark/50'}
      `}
      animate={isActive ? { 
        boxShadow: [`0 0 20px ${isAttacker ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.3)'}`, 
                    `0 0 30px ${isAttacker ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.5)'}`,
                    `0 0 20px ${isAttacker ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.3)'}`]
      } : {}}
      transition={{ duration: 1.5, repeat: isActive ? Infinity : 0 }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className={`w-6 h-6 text-${color}`} />
          <span className={`font-bold text-${color} uppercase`}>
            {player}
          </span>
        </div>
        {isActive && (
          <span className="px-2 py-1 bg-white/10 rounded text-xs text-white animate-pulse">
            Active
          </span>
        )}
      </div>
      
      <div className="text-3xl font-mono font-bold text-white mb-2">
        {score.toLocaleString()}
      </div>
      
      <div className="flex gap-4 text-xs">
        <div className="flex items-center gap-1">
          {isAttacker ? (
            <>
              <Target className="w-3 h-3 text-red-400" />
              <span className="text-gray-400">Compromised: {stats.compromised}</span>
            </>
          ) : (
            <>
              <CheckCircle2 className="w-3 h-3 text-green-400" />
              <span className="text-gray-400">Secured: {stats.secured}</span>
            </>
          )}
        </div>
      </div>
    </motion.div>
  )
}

function GameInfo({ gameState }: GameInfoProps) {
  if (!gameState) return null
  
  // Use snake_case properties from API
  const currentPlayer = gameState.current_player?.toLowerCase() as Player | undefined
  const turn = gameState.turn_number || 0
  const scores = {
    attacker: gameState.attacker_score || 0,
    defender: gameState.defender_score || 0
  }
  const gameOver = gameState.game_over || false
  const winner = gameState.winner?.toLowerCase()
  
  // Calculate stats from nodes array
  const nodes = gameState.nodes || []
  const compromisedCount = nodes.filter(n => n.is_compromised).length
  const securedCount = nodes.filter(n => !n.is_compromised).length
  const totalNodes = nodes.length
  
  return (
    <motion.div 
      className="glass-card rounded-xl p-4 border border-cyber-primary/30"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Game Over Banner */}
      {gameOver && winner && (
        <motion.div 
          className={`
            mb-4 p-4 rounded-lg flex items-center gap-3
            ${winner === 'attacker' ? 'bg-red-500/20 border border-red-500' : 'bg-green-500/20 border border-green-500'}
          `}
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
        >
          <Trophy className={`w-8 h-8 ${winner === 'attacker' ? 'text-red-400' : 'text-green-400'}`} />
          <div>
            <div className="font-bold text-white text-lg">Game Over!</div>
            <div className={`${winner === 'attacker' ? 'text-red-400' : 'text-green-400'}`}>
              {winner.charAt(0).toUpperCase() + winner.slice(1)} Wins!
            </div>
          </div>
        </motion.div>
      )}
      
      {/* Turn Counter */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Clock className="w-5 h-5 text-cyber-primary" />
          <span className="text-gray-400">Turn</span>
          <span className="font-mono font-bold text-cyber-primary text-xl">{turn}</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-gray-400">Nodes:</span>
          <span className="text-white font-mono">{totalNodes}</span>
        </div>
      </div>
      
      {/* Player Scores */}
      <div className="flex gap-4 mb-4">
        <PlayerScore 
          player="attacker" 
          score={scores?.attacker || 0}
          isActive={currentPlayer === 'attacker'}
          stats={{ compromised: compromisedCount, secured: 0 }}
        />
        <PlayerScore 
          player="defender" 
          score={scores?.defender || 0}
          isActive={currentPlayer === 'defender'}
          stats={{ compromised: 0, secured: securedCount }}
        />
      </div>
      
      {/* Network Stats */}
      <div className="grid grid-cols-3 gap-4 p-3 rounded-lg bg-cyber-dark/50 border border-gray-700">
        <div className="text-center">
          <div className="text-2xl font-mono font-bold text-red-400">{compromisedCount}</div>
          <div className="text-xs text-gray-400">Compromised</div>
        </div>
        <div className="text-center border-x border-gray-700">
          <div className="text-2xl font-mono font-bold text-yellow-400">
            {totalNodes - compromisedCount - securedCount}
          </div>
          <div className="text-xs text-gray-400">Neutral</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-mono font-bold text-green-400">{securedCount}</div>
          <div className="text-xs text-gray-400">Secured</div>
        </div>
      </div>
      
      {/* Progress Bar */}
      <div className="mt-4">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Attacker Progress</span>
          <span>{Math.round((compromisedCount / Math.max(totalNodes, 1)) * 100)}%</span>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <motion.div 
            className="h-full bg-gradient-to-r from-red-500 to-orange-500"
            initial={{ width: 0 }}
            animate={{ width: `${(compromisedCount / Math.max(totalNodes, 1)) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>
      
      {/* Warning if critical nodes compromised */}
      {compromisedCount > totalNodes * 0.5 && !gameOver && (
        <motion.div 
          className="mt-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/50 flex items-center gap-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <AlertTriangle className="w-5 h-5 text-yellow-400" />
          <span className="text-yellow-400 text-sm">Critical breach level reached!</span>
        </motion.div>
      )}
    </motion.div>
  )
}

export default GameInfo
