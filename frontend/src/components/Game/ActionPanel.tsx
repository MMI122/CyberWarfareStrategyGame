import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Sword, 
  Shield, 
  Search, 
  Bug, 
  Lock, 
  Zap,
  Network,
  Eye,
  RefreshCw,
  ChevronRight
} from 'lucide-react'
import { GameAction, ActionType, GameState } from '@/types'

interface ActionPanelProps {
  gameState: GameState
  selectedNode: string | null
  onActionSelect: (action: GameAction) => void
  isPlayerTurn: boolean
  isProcessing: boolean
}

interface ActionOption {
  type: ActionType
  icon: typeof Sword
  label: string
  description: string
  color: string
  bgColor: string
  isAttacker: boolean
}

const actionOptions: ActionOption[] = [
  // Attacker actions
  {
    type: ActionType.SCAN,
    icon: Search,
    label: 'Scan Network',
    description: 'Discover vulnerabilities and network structure',
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-400/10',
    isAttacker: true
  },
  {
    type: ActionType.EXPLOIT,
    icon: Bug,
    label: 'Exploit Vulnerability',
    description: 'Attempt to exploit a discovered vulnerability',
    color: 'text-red-400',
    bgColor: 'bg-red-400/10',
    isAttacker: true
  },
  {
    type: ActionType.PIVOT,
    icon: Network,
    label: 'Lateral Movement',
    description: 'Move to adjacent compromised nodes',
    color: 'text-orange-400',
    bgColor: 'bg-orange-400/10',
    isAttacker: true
  },
  {
    type: ActionType.INSTALL_BACKDOOR,
    icon: Zap,
    label: 'Install Backdoor',
    description: 'Maintain persistent access on compromised node',
    color: 'text-purple-400',
    bgColor: 'bg-purple-400/10',
    isAttacker: true
  },
  {
    type: ActionType.EXFILTRATE,
    icon: Sword,
    label: 'Exfiltrate Data',
    description: 'Extract valuable data from node',
    color: 'text-pink-400',
    bgColor: 'bg-pink-400/10',
    isAttacker: true
  },
  // Defender actions
  {
    type: ActionType.PATCH,
    icon: Shield,
    label: 'Patch System',
    description: 'Fix vulnerabilities on a node',
    color: 'text-green-400',
    bgColor: 'bg-green-400/10',
    isAttacker: false
  },
  {
    type: ActionType.ISOLATE,
    icon: Lock,
    label: 'Isolate Node',
    description: 'Disconnect a compromised node from network',
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-400/10',
    isAttacker: false
  },
  {
    type: ActionType.MONITOR,
    icon: Eye,
    label: 'Deploy Monitor',
    description: 'Set up detection on a node',
    color: 'text-blue-400',
    bgColor: 'bg-blue-400/10',
    isAttacker: false
  },
  {
    type: ActionType.RESTORE,
    icon: RefreshCw,
    label: 'Restore System',
    description: 'Restore a compromised node',
    color: 'text-teal-400',
    bgColor: 'bg-teal-400/10',
    isAttacker: false
  },
]

function ActionPanel({ 
  gameState, 
  selectedNode, 
  onActionSelect, 
  isPlayerTurn,
  isProcessing 
}: ActionPanelProps) {
  const [selectedAction, setSelectedAction] = useState<ActionType | null>(null)
  const [targetNode, setTargetNode] = useState<string | null>(null)
  
  // Reset selection when node changes
  useEffect(() => {
    setSelectedAction(null)
    setTargetNode(selectedNode)
  }, [selectedNode])
  
  const isAttackerTurn = gameState?.current_player === 'ATTACKER'
  const availableActions = actionOptions.filter(a => a.isAttacker === isAttackerTurn)
  
  const handleActionClick = (actionType: ActionType) => {
    setSelectedAction(actionType)
  }
  
  const handleConfirmAction = () => {
    if (!selectedAction || !targetNode) return
    
    const action: GameAction = {
      type: selectedAction,
      target_node: parseInt(targetNode),
      targetNodeId: targetNode,
    }
    
    onActionSelect(action)
    setSelectedAction(null)
  }
  
  const canExecuteAction = selectedAction && targetNode && isPlayerTurn && !isProcessing
  
  // Find the selected node in the nodes array
  const selectedNodeInfo = gameState?.nodes?.find(n => String(n.id) === selectedNode)
  
  return (
    <motion.div 
      className="glass-card rounded-xl p-4 border border-cyber-primary/30"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-white font-mono flex items-center gap-2">
          {isAttackerTurn ? (
            <>
              <Sword className="w-5 h-5 text-red-400" />
              Attack Actions
            </>
          ) : (
            <>
              <Shield className="w-5 h-5 text-green-400" />
              Defense Actions
            </>
          )}
        </h3>
        <div className={`px-3 py-1 rounded-full text-xs font-mono ${
          isPlayerTurn ? 'bg-cyber-primary/20 text-cyber-primary' : 'bg-gray-700 text-gray-400'
        }`}>
          {isPlayerTurn ? 'Your Turn' : 'AI Thinking...'}
        </div>
      </div>
      
      {/* Selected Node Info */}
      {selectedNode && selectedNodeInfo && (
        <div className="mb-4 p-3 rounded-lg bg-cyber-dark/50 border border-cyber-primary/20">
          <div className="text-xs text-gray-400 mb-1">Selected Target</div>
          <div className="text-cyber-primary font-mono font-semibold">
            {selectedNodeInfo.name}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Type: {selectedNodeInfo.node_type}
          </div>
        </div>
      )}
      
      {/* Action Buttons */}
      <div className="space-y-2 mb-4">
        <AnimatePresence>
          {availableActions.map((action, index) => (
            <motion.button
              key={action.type}
              onClick={() => handleActionClick(action.type)}
              disabled={!isPlayerTurn || isProcessing}
              className={`
                w-full p-3 rounded-lg border-2 text-left transition-all
                flex items-center gap-3
                ${selectedAction === action.type 
                  ? `border-current ${action.color} ${action.bgColor}` 
                  : 'border-gray-700 hover:border-gray-600'}
                ${(!isPlayerTurn || isProcessing) && 'opacity-50 cursor-not-allowed'}
              `}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, delay: index * 0.05 }}
              whileHover={isPlayerTurn && !isProcessing ? { scale: 1.02 } : {}}
              whileTap={isPlayerTurn && !isProcessing ? { scale: 0.98 } : {}}
            >
              <div className={`p-2 rounded-lg ${action.bgColor}`}>
                <action.icon className={`w-5 h-5 ${action.color}`} />
              </div>
              <div className="flex-1">
                <div className="font-semibold text-white text-sm">{action.label}</div>
                <div className="text-xs text-gray-400">{action.description}</div>
              </div>
              {selectedAction === action.type && (
                <ChevronRight className={`w-5 h-5 ${action.color}`} />
              )}
            </motion.button>
          ))}
        </AnimatePresence>
      </div>
      
      {/* Confirm Button */}
      <motion.button
        onClick={handleConfirmAction}
        disabled={!canExecuteAction}
        className={`
          w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2
          transition-all duration-300
          ${canExecuteAction 
            ? 'bg-gradient-to-r from-cyber-primary to-cyber-secondary text-black hover:shadow-cyber-glow' 
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'}
        `}
        whileHover={canExecuteAction ? { scale: 1.02 } : {}}
        whileTap={canExecuteAction ? { scale: 0.98 } : {}}
      >
        {isProcessing ? (
          <>
            <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <Zap className="w-5 h-5" />
            Execute Action
          </>
        )}
      </motion.button>
      
      {/* Help text */}
      {!selectedNode && (
        <p className="text-xs text-gray-500 text-center mt-3">
          Click on a node in the network to select a target
        </p>
      )}
    </motion.div>
  )
}

export default ActionPanel
