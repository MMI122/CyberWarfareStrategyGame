import { motion } from 'framer-motion'
import { 
  Settings as SettingsIcon, 
  Monitor, 
  Palette,
  Brain,
  Network,
  Save,
  RotateCcw
} from 'lucide-react'
import { useSettingsStore } from '@/store'
import { Button, Card, CardHeader, CardTitle, CardContent, useToast } from '@/components/UI'
import { TopologyType, DifficultyLevel, AIType, PlayMode } from '@/types'

function SettingsPage() {
  const { addToast } = useToast()
  const { settings, updateSettings, resetSettings } = useSettingsStore()
  
  const handleSave = () => {
    addToast('success', 'Settings saved!')
  }
  
  const handleReset = () => {
    resetSettings()
    addToast('info', 'Settings reset to defaults')
  }
  
  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <div className="flex items-center justify-center gap-3 mb-4">
          <SettingsIcon className="w-12 h-12 text-cyber-primary" />
        </div>
        <h1 className="text-4xl font-bold text-white font-mono mb-2">
          Settings
        </h1>
        <p className="text-gray-400">
          Configure game options, AI parameters, and visual preferences
        </p>
      </motion.div>
      
      <div className="space-y-6">
        {/* Game Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Monitor className="w-5 h-5 text-cyber-primary" />
              Game Settings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Play Mode */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Play Mode</label>
                <div className="grid grid-cols-3 gap-3">
                  {(['vs_ai', 'ai_vs_ai', 'pvp'] as PlayMode[]).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => updateSettings({ playMode: mode })}
                      className={`
                        p-3 rounded-lg border-2 text-center transition-all
                        ${settings.playMode === mode 
                          ? 'border-cyber-primary bg-cyber-primary/20 text-cyber-primary' 
                          : 'border-gray-700 text-gray-400 hover:border-gray-600'}
                      `}
                    >
                      <div className="font-semibold capitalize">
                        {mode.replace('_', ' ')}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Topology */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Default Network Topology</label>
                <div className="grid grid-cols-4 gap-3">
                  {(['mesh', 'star', 'tree', 'hybrid'] as TopologyType[]).map((topology) => (
                    <button
                      key={topology}
                      onClick={() => updateSettings({ topology })}
                      className={`
                        p-3 rounded-lg border-2 text-center transition-all
                        ${settings.topology === topology 
                          ? 'border-cyber-secondary bg-cyber-secondary/20 text-cyber-secondary' 
                          : 'border-gray-700 text-gray-400 hover:border-gray-600'}
                      `}
                    >
                      <Network className="w-5 h-5 mx-auto mb-1" />
                      <div className="text-sm capitalize">{topology}</div>
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Difficulty */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Difficulty</label>
                <div className="grid grid-cols-4 gap-3">
                  {(['easy', 'medium', 'hard', 'expert'] as DifficultyLevel[]).map((difficulty) => (
                    <button
                      key={difficulty}
                      onClick={() => updateSettings({ difficulty })}
                      className={`
                        p-3 rounded-lg border-2 text-center transition-all capitalize
                        ${settings.difficulty === difficulty 
                          ? 'border-yellow-400 bg-yellow-400/20 text-yellow-400' 
                          : 'border-gray-700 text-gray-400 hover:border-gray-600'}
                      `}
                    >
                      {difficulty}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* AI Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              AI Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* AI Type */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Default AI Type</label>
                <div className="grid grid-cols-2 gap-4">
                  {(['minmax', 'deep_rl'] as AIType[]).map((aiType) => (
                    <button
                      key={aiType}
                      onClick={() => updateSettings({ aiType })}
                      className={`
                        p-4 rounded-lg border-2 text-left transition-all
                        ${settings.aiType === aiType 
                          ? 'border-purple-400 bg-purple-400/20' 
                          : 'border-gray-700 hover:border-gray-600'}
                      `}
                    >
                      <div className={`font-bold mb-1 ${settings.aiType === aiType ? 'text-purple-400' : 'text-white'}`}>
                        {aiType === 'minmax' ? 'MinMax AI' : 'Deep RL AI'}
                      </div>
                      <div className="text-xs text-gray-400">
                        {aiType === 'minmax' 
                          ? 'Alpha-Beta pruning, transposition tables, killer moves' 
                          : 'PPO + MCTS neural network approach'}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
              
              {/* AI Difficulty */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  AI Search Depth / Difficulty
                </label>
                <input
                  type="range"
                  min="1"
                  max="4"
                  value={
                    settings.aiDifficulty === 'easy' ? 1 :
                    settings.aiDifficulty === 'medium' ? 2 :
                    settings.aiDifficulty === 'hard' ? 3 : 4
                  }
                  onChange={(e) => {
                    const val = parseInt(e.target.value)
                    const difficulty = val === 1 ? 'easy' : val === 2 ? 'medium' : val === 3 ? 'hard' : 'expert'
                    updateSettings({ aiDifficulty: difficulty })
                  }}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Easy</span>
                  <span>Medium</span>
                  <span>Hard</span>
                  <span>Expert</span>
                </div>
              </div>
              
              {/* Thinking Time */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  AI Time Limit: {settings.aiTimeLimit / 1000}s
                </label>
                <input
                  type="range"
                  min="1000"
                  max="30000"
                  step="1000"
                  value={settings.aiTimeLimit}
                  onChange={(e) => updateSettings({ aiTimeLimit: parseInt(e.target.value) })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1s</span>
                  <span>30s</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Visual Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Palette className="w-5 h-5 text-cyan-400" />
              Visual & Audio
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Animation Speed */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Animation Speed</label>
                <div className="grid grid-cols-3 gap-3">
                  {(['slow', 'normal', 'fast'] as const).map((speed) => (
                    <button
                      key={speed}
                      onClick={() => updateSettings({ animationSpeed: speed })}
                      className={`
                        p-3 rounded-lg border-2 text-center transition-all capitalize
                        ${settings.animationSpeed === speed 
                          ? 'border-cyan-400 bg-cyan-400/20 text-cyan-400' 
                          : 'border-gray-700 text-gray-400 hover:border-gray-600'}
                      `}
                    >
                      {speed}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Sound */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-white">Sound Effects</div>
                  <div className="text-sm text-gray-400">Enable audio feedback</div>
                </div>
                <button
                  onClick={() => updateSettings({ soundEnabled: !settings.soundEnabled })}
                  className={`
                    w-14 h-8 rounded-full transition-all relative
                    ${settings.soundEnabled ? 'bg-cyber-primary' : 'bg-gray-700'}
                  `}
                >
                  <div className={`
                    absolute top-1 w-6 h-6 rounded-full bg-white transition-all
                    ${settings.soundEnabled ? 'left-7' : 'left-1'}
                  `} />
                </button>
              </div>
              
              {/* Particles */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-white">Background Particles</div>
                  <div className="text-sm text-gray-400">Show animated particles</div>
                </div>
                <button
                  onClick={() => updateSettings({ showParticles: !settings.showParticles })}
                  className={`
                    w-14 h-8 rounded-full transition-all relative
                    ${settings.showParticles ? 'bg-cyber-primary' : 'bg-gray-700'}
                  `}
                >
                  <div className={`
                    absolute top-1 w-6 h-6 rounded-full bg-white transition-all
                    ${settings.showParticles ? 'left-7' : 'left-1'}
                  `} />
                </button>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Action Buttons */}
        <div className="flex gap-4 justify-end">
          <Button
            variant="ghost"
            onClick={handleReset}
            leftIcon={<RotateCcw className="w-4 h-4" />}
          >
            Reset to Defaults
          </Button>
          <Button
            onClick={handleSave}
            leftIcon={<Save className="w-4 h-4" />}
          >
            Save Settings
          </Button>
        </div>
      </div>
    </div>
  )
}

export default SettingsPage
