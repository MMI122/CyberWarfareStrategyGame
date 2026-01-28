import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Shield, 
  Sword, 
  Network, 
  Cpu, 
  Play, 
  Zap,
  Target,
  Brain,
  ChevronRight,
  Layers
} from 'lucide-react'
import { useGameStore, useSettingsStore } from '@/store'
import { gameApi } from '@/utils/api'
import { TopologyType, DifficultyLevel, AIType } from '@/types'

const topologyOptions: { value: TopologyType; label: string; description: string }[] = [
  { value: 'mesh', label: 'Mesh Network', description: 'Highly connected, complex topology' },
  { value: 'star', label: 'Star Network', description: 'Central hub with peripheral nodes' },
  { value: 'tree', label: 'Tree Network', description: 'Hierarchical structure' },
  { value: 'hybrid', label: 'Hybrid Network', description: 'Mixed topology types' },
]

const difficultyOptions: { value: DifficultyLevel; label: string; color: string }[] = [
  { value: 'easy', label: 'Easy', color: 'text-green-400' },
  { value: 'medium', label: 'Medium', color: 'text-yellow-400' },
  { value: 'hard', label: 'Hard', color: 'text-orange-400' },
  { value: 'expert', label: 'Expert', color: 'text-red-400' },
]

const aiOptions: { value: AIType; label: string; icon: typeof Cpu; description: string }[] = [
  { 
    value: 'minmax', 
    label: 'MinMax AI', 
    icon: Target,
    description: 'Alpha-Beta pruning with advanced heuristics'
  },
  { 
    value: 'deep_rl', 
    label: 'Deep RL AI', 
    icon: Brain,
    description: 'PPO + MCTS neural network approach'
  },
]

const features = [
  {
    icon: Network,
    title: 'Complex Network Topology',
    description: 'Dynamic cyber networks with realistic vulnerabilities and defense mechanisms'
  },
  {
    icon: Cpu,
    title: 'Advanced AI Agents',
    description: 'MinMax with Alpha-Beta pruning and Deep RL with PPO + MCTS'
  },
  {
    icon: Shield,
    title: 'Strategic Gameplay',
    description: 'Balance attack and defense in turn-based cyber warfare'
  },
  {
    icon: Zap,
    title: 'Research Ready',
    description: 'Built for AI research with comprehensive metrics and analysis'
  },
]

function HomePage() {
  const navigate = useNavigate()
  const { setGameState, setGameId, setLoading, setError } = useGameStore()
  const { settings, updateSettings } = useSettingsStore()
  
  const [isCreating, setIsCreating] = useState(false)
  const [selectedTopology, setSelectedTopology] = useState<TopologyType>(settings.topology)
  const [selectedDifficulty, setSelectedDifficulty] = useState<DifficultyLevel>(settings.difficulty)
  const [selectedAI, setSelectedAI] = useState<AIType>(settings.aiType)
  
  const handleCreateGame = async () => {
    setIsCreating(true)
    setLoading(true)
    setError(null)
    
    try {
      // Update settings
      updateSettings({
        topology: selectedTopology,
        difficulty: selectedDifficulty,
        aiType: selectedAI,
      })
      
      // Create game via API
      const response = await gameApi.createGame(selectedTopology, selectedDifficulty)
      
      setGameId(response.game_id)
      setGameState(response)
      
      // Navigate to game
      navigate('/game')
    } catch (err) {
      console.error('Failed to create game:', err)
      setError('Failed to create game. Make sure the backend server is running.')
    } finally {
      setIsCreating(false)
      setLoading(false)
    }
  }
  
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="flex items-center justify-center gap-4 mb-6">
              <Shield className="w-16 h-16 text-cyber-primary cyber-glow" />
              <Sword className="w-16 h-16 text-cyber-accent cyber-glow" />
            </div>
            <h1 className="text-5xl md:text-7xl font-bold mb-4 font-mono">
              <span className="text-gradient">CYBER WARFARE</span>
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              A turn-based strategy game pitting advanced AI agents against each other
              in simulated cyber warfare scenarios
            </p>
          </motion.div>
        </div>
        
        {/* Features Grid */}
        <motion.div 
          className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="glass-card p-6 rounded-xl border border-cyber-primary/20 hover:border-cyber-primary/50 transition-all"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.1 * index }}
              whileHover={{ scale: 1.02 }}
            >
              <feature.icon className="w-10 h-10 text-cyber-primary mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-gray-400 text-sm">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </section>
      
      {/* Game Setup Section */}
      <section className="container mx-auto px-4 pb-16">
        <motion.div
          className="max-w-4xl mx-auto glass-card rounded-2xl p-8 border border-cyber-primary/30"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <h2 className="text-3xl font-bold text-white mb-8 text-center font-mono">
            Start New Game
          </h2>
          
          {/* Topology Selection */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-cyber-primary mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Network Topology
            </h3>
            <div className="grid sm:grid-cols-2 gap-4">
              {topologyOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setSelectedTopology(option.value)}
                  className={`
                    p-4 rounded-lg border-2 text-left transition-all
                    ${selectedTopology === option.value 
                      ? 'border-cyber-primary bg-cyber-primary/20' 
                      : 'border-gray-700 hover:border-cyber-primary/50'}
                  `}
                >
                  <div className="font-semibold text-white">{option.label}</div>
                  <div className="text-sm text-gray-400">{option.description}</div>
                </button>
              ))}
            </div>
          </div>
          
          {/* Difficulty Selection */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-cyber-primary mb-4 flex items-center gap-2">
              <Target className="w-5 h-5" />
              Difficulty Level
            </h3>
            <div className="flex flex-wrap gap-3">
              {difficultyOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setSelectedDifficulty(option.value)}
                  className={`
                    px-6 py-3 rounded-lg border-2 font-semibold transition-all
                    ${selectedDifficulty === option.value 
                      ? `border-current bg-current/20 ${option.color}` 
                      : 'border-gray-700 text-gray-400 hover:border-gray-500'}
                  `}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          
          {/* AI Selection */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-cyber-primary mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5" />
              AI Opponent
            </h3>
            <div className="grid sm:grid-cols-2 gap-4">
              {aiOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setSelectedAI(option.value)}
                  className={`
                    p-4 rounded-lg border-2 text-left transition-all flex items-start gap-4
                    ${selectedAI === option.value 
                      ? 'border-cyber-secondary bg-cyber-secondary/20' 
                      : 'border-gray-700 hover:border-cyber-secondary/50'}
                  `}
                >
                  <option.icon className={`w-8 h-8 ${selectedAI === option.value ? 'text-cyber-secondary' : 'text-gray-500'}`} />
                  <div>
                    <div className="font-semibold text-white">{option.label}</div>
                    <div className="text-sm text-gray-400">{option.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
          
          {/* Start Button */}
          <motion.button
            onClick={handleCreateGame}
            disabled={isCreating}
            className={`
              w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-3
              transition-all duration-300
              ${isCreating 
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed' 
                : 'bg-gradient-to-r from-cyber-primary to-cyber-secondary text-black hover:shadow-[0_0_30px_rgba(0,255,136,0.5)]'}
            `}
            whileHover={!isCreating ? { scale: 1.02 } : {}}
            whileTap={!isCreating ? { scale: 0.98 } : {}}
          >
            {isCreating ? (
              <>
                <div className="w-6 h-6 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                Initializing...
              </>
            ) : (
              <>
                <Play className="w-6 h-6" />
                Launch Cyber Warfare
                <ChevronRight className="w-6 h-6" />
              </>
            )}
          </motion.button>
        </motion.div>
      </section>
    </div>
  )
}

export default HomePage
