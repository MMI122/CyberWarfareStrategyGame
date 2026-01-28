import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  FlaskConical, 
  Play, 
  BarChart3, 
  Brain, 
  Target,
  Clock,
  TrendingUp,
  Download,
  Swords
} from 'lucide-react'
import { Button, Card, CardHeader, CardTitle, CardContent, useToast } from '@/components/UI'
import { gameApi, aiApi } from '@/utils/api'
import { AIType } from '@/types'

interface BenchmarkResult {
  aiType: AIType
  wins: number
  losses: number
  draws: number
  avgMoveTime: number
  avgScore: number
  totalGames: number
}

interface TournamentMatch {
  attacker: AIType
  defender: AIType
  winner: string | null
  turns: number
  attackerScore: number
  defenderScore: number
}

function ResearchPage() {
  const { addToast } = useToast()
  
  const [isRunningBenchmark, setIsRunningBenchmark] = useState(false)
  const [isRunningTournament, setIsRunningTournament] = useState(false)
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([])
  const [tournamentResults, setTournamentResults] = useState<TournamentMatch[]>([])
  const [selectedAI1, setSelectedAI1] = useState<AIType>('minmax')
  const [selectedAI2, setSelectedAI2] = useState<AIType>('deep_rl')
  const [numGames, setNumGames] = useState(10)
  const [progress, setProgress] = useState(0)
  
  // Run benchmark for a single AI
  const runBenchmark = async (aiType: AIType): Promise<BenchmarkResult> => {
    let wins = 0
    let losses = 0
    let draws = 0
    let totalMoveTime = 0
    let totalScore = 0
    let moveCount = 0
    
    for (let i = 0; i < numGames; i++) {
      try {
        // Create a new game
        const game = await gameApi.createGame('mesh', 'medium')
        let state = game.state
        let gameOver = false
        
        while (!gameOver && state.turn < 100) {
          const startTime = Date.now()
          
          // Get AI move
          let result
          if (aiType === 'minmax') {
            result = await aiApi.getMinmaxMove(game.game_id, { depth: 4, timeLimit: 5000 })
          } else {
            result = await aiApi.getDeepRLMove(game.game_id)
          }
          
          totalMoveTime += Date.now() - startTime
          moveCount++
          
          if (result.action) {
            const actionResult = await gameApi.performAction(game.game_id, result.action)
            state = actionResult.state
          }
          
          if (state.gameOver) {
            gameOver = true
            if (state.winner === 'attacker') {
              if (aiType === 'minmax') wins++
              else losses++
            } else if (state.winner === 'defender') {
              if (aiType === 'deep_rl') wins++
              else losses++
            } else {
              draws++
            }
            totalScore += state.scores?.attacker || 0
          }
        }
        
        if (!gameOver) draws++
        
        setProgress(((i + 1) / numGames) * 100)
      } catch (err) {
        console.error('Benchmark game failed:', err)
      }
    }
    
    return {
      aiType,
      wins,
      losses,
      draws,
      avgMoveTime: moveCount > 0 ? totalMoveTime / moveCount : 0,
      avgScore: totalScore / numGames,
      totalGames: numGames
    }
  }
  
  // Run full benchmark
  const handleRunBenchmark = async () => {
    setIsRunningBenchmark(true)
    setProgress(0)
    setBenchmarkResults([])
    
    try {
      addToast('info', 'Starting benchmark...')
      
      const minmaxResult = await runBenchmark('minmax')
      setBenchmarkResults(prev => [...prev, minmaxResult])
      
      const deepRLResult = await runBenchmark('deep_rl')
      setBenchmarkResults(prev => [...prev, deepRLResult])
      
      addToast('success', 'Benchmark complete!')
    } catch (err) {
      addToast('error', 'Benchmark failed')
    } finally {
      setIsRunningBenchmark(false)
      setProgress(100)
    }
  }
  
  // Run tournament between two AIs
  const handleRunTournament = async () => {
    setIsRunningTournament(true)
    setProgress(0)
    setTournamentResults([])
    
    try {
      addToast('info', `Starting tournament: ${selectedAI1} vs ${selectedAI2}`)
      
      for (let i = 0; i < numGames; i++) {
        const game = await gameApi.createGame('hybrid', 'hard')
        let state = game.state
        
        const match: TournamentMatch = {
          attacker: selectedAI1,
          defender: selectedAI2,
          winner: null,
          turns: 0,
          attackerScore: 0,
          defenderScore: 0
        }
        
        while (!state.gameOver && state.turn < 100) {
          const currentAI = state.currentPlayer === 'attacker' ? selectedAI1 : selectedAI2
          
          let result
          if (currentAI === 'minmax') {
            result = await aiApi.getMinmaxMove(game.game_id, { depth: 4, timeLimit: 5000 })
          } else {
            result = await aiApi.getDeepRLMove(game.game_id)
          }
          
          if (result.action) {
            const actionResult = await gameApi.performAction(game.game_id, result.action)
            state = actionResult.state
          }
          
          match.turns = state.turn
        }
        
        match.winner = state.winner
        match.attackerScore = state.scores?.attacker || 0
        match.defenderScore = state.scores?.defender || 0
        
        setTournamentResults(prev => [...prev, match])
        setProgress(((i + 1) / numGames) * 100)
      }
      
      addToast('success', 'Tournament complete!')
    } catch (err) {
      addToast('error', 'Tournament failed')
    } finally {
      setIsRunningTournament(false)
    }
  }
  
  // Export results as JSON
  const handleExportResults = () => {
    const data = {
      timestamp: new Date().toISOString(),
      benchmarkResults,
      tournamentResults,
      settings: {
        numGames,
        ai1: selectedAI1,
        ai2: selectedAI2
      }
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `cyber-warfare-research-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    
    addToast('success', 'Results exported!')
  }
  
  // Calculate tournament statistics
  const tournamentStats = {
    ai1Wins: tournamentResults.filter(r => 
      (r.attacker === selectedAI1 && r.winner === 'attacker') ||
      (r.defender === selectedAI1 && r.winner === 'defender')
    ).length,
    ai2Wins: tournamentResults.filter(r => 
      (r.attacker === selectedAI2 && r.winner === 'attacker') ||
      (r.defender === selectedAI2 && r.winner === 'defender')
    ).length,
    draws: tournamentResults.filter(r => r.winner === null).length,
    avgTurns: tournamentResults.length > 0 
      ? Math.round(tournamentResults.reduce((sum, r) => sum + r.turns, 0) / tournamentResults.length)
      : 0
  }
  
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <div className="flex items-center justify-center gap-3 mb-4">
          <FlaskConical className="w-12 h-12 text-cyber-secondary" />
        </div>
        <h1 className="text-4xl font-bold text-white font-mono mb-2">
          AI Research Lab
        </h1>
        <p className="text-gray-400 max-w-2xl mx-auto">
          Benchmark and compare AI agents. Run tournaments, analyze performance metrics,
          and export data for academic research.
        </p>
      </motion.div>
      
      {/* Configuration */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* Benchmark Config */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-cyber-primary" />
              Benchmark Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Number of Games</label>
                <input
                  type="number"
                  value={numGames}
                  onChange={(e) => setNumGames(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full px-4 py-2 rounded-lg bg-cyber-dark border border-gray-700 text-white font-mono focus:border-cyber-primary focus:outline-none"
                  min="1"
                  max="100"
                />
              </div>
              
              <Button
                onClick={handleRunBenchmark}
                disabled={isRunningBenchmark}
                isLoading={isRunningBenchmark}
                className="w-full"
                leftIcon={<Play className="w-4 h-4" />}
              >
                Run Benchmark
              </Button>
            </div>
          </CardContent>
        </Card>
        
        {/* Tournament Config */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Swords className="w-5 h-5 text-cyber-accent" />
              Tournament Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">AI 1 (Attacker)</label>
                  <select
                    value={selectedAI1}
                    onChange={(e) => setSelectedAI1(e.target.value as AIType)}
                    className="w-full px-4 py-2 rounded-lg bg-cyber-dark border border-gray-700 text-white font-mono focus:border-cyber-primary focus:outline-none"
                  >
                    <option value="minmax">MinMax</option>
                    <option value="deep_rl">Deep RL</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">AI 2 (Defender)</label>
                  <select
                    value={selectedAI2}
                    onChange={(e) => setSelectedAI2(e.target.value as AIType)}
                    className="w-full px-4 py-2 rounded-lg bg-cyber-dark border border-gray-700 text-white font-mono focus:border-cyber-primary focus:outline-none"
                  >
                    <option value="minmax">MinMax</option>
                    <option value="deep_rl">Deep RL</option>
                  </select>
                </div>
              </div>
              
              <Button
                onClick={handleRunTournament}
                disabled={isRunningTournament}
                isLoading={isRunningTournament}
                variant="secondary"
                className="w-full"
                leftIcon={<Swords className="w-4 h-4" />}
              >
                Run Tournament
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Progress Bar */}
      {(isRunningBenchmark || isRunningTournament) && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-8"
        >
          <div className="flex justify-between text-sm text-gray-400 mb-2">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-cyber-primary to-cyber-secondary"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
            />
          </div>
        </motion.div>
      )}
      
      {/* Results */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* Benchmark Results */}
        {benchmarkResults.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                Benchmark Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {benchmarkResults.map((result, index) => (
                  <div key={index} className="p-4 rounded-lg bg-cyber-dark/50 border border-gray-700">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        {result.aiType === 'minmax' ? (
                          <Target className="w-5 h-5 text-yellow-400" />
                        ) : (
                          <Brain className="w-5 h-5 text-purple-400" />
                        )}
                        <span className="font-mono font-bold text-white uppercase">
                          {result.aiType}
                        </span>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <div className="text-2xl font-mono font-bold text-green-400">{result.wins}</div>
                        <div className="text-xs text-gray-400">Wins</div>
                      </div>
                      <div>
                        <div className="text-2xl font-mono font-bold text-red-400">{result.losses}</div>
                        <div className="text-xs text-gray-400">Losses</div>
                      </div>
                      <div>
                        <div className="text-2xl font-mono font-bold text-gray-400">{result.draws}</div>
                        <div className="text-xs text-gray-400">Draws</div>
                      </div>
                    </div>
                    
                    <div className="mt-3 pt-3 border-t border-gray-700 flex justify-between text-sm">
                      <div className="flex items-center gap-1 text-gray-400">
                        <Clock className="w-4 h-4" />
                        Avg: {result.avgMoveTime.toFixed(0)}ms
                      </div>
                      <div className="text-cyber-primary">
                        Score: {result.avgScore.toFixed(1)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* Tournament Results */}
        {tournamentResults.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Swords className="w-5 h-5 text-cyber-accent" />
                Tournament Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* Stats Summary */}
              <div className="grid grid-cols-4 gap-4 mb-4 p-4 rounded-lg bg-cyber-dark/50 border border-gray-700">
                <div className="text-center">
                  <div className="text-xl font-mono font-bold text-yellow-400">{tournamentStats.ai1Wins}</div>
                  <div className="text-xs text-gray-400">{selectedAI1} Wins</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-mono font-bold text-purple-400">{tournamentStats.ai2Wins}</div>
                  <div className="text-xs text-gray-400">{selectedAI2} Wins</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-mono font-bold text-gray-400">{tournamentStats.draws}</div>
                  <div className="text-xs text-gray-400">Draws</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-mono font-bold text-cyan-400">{tournamentStats.avgTurns}</div>
                  <div className="text-xs text-gray-400">Avg Turns</div>
                </div>
              </div>
              
              {/* Match History */}
              <div className="max-h-64 overflow-y-auto space-y-2">
                {tournamentResults.map((match, index) => (
                  <div
                    key={index}
                    className={`
                      p-3 rounded-lg border text-sm flex items-center justify-between
                      ${match.winner === 'attacker' 
                        ? 'bg-red-500/10 border-red-500/30' 
                        : match.winner === 'defender'
                        ? 'bg-green-500/10 border-green-500/30'
                        : 'bg-gray-500/10 border-gray-500/30'}
                    `}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-gray-400">#{index + 1}</span>
                      <span className="font-mono text-white">
                        {match.attacker} vs {match.defender}
                      </span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="text-gray-400">{match.turns} turns</span>
                      <span className={`font-bold ${
                        match.winner === 'attacker' ? 'text-red-400' :
                        match.winner === 'defender' ? 'text-green-400' : 'text-gray-400'
                      }`}>
                        {match.winner ? `${match.winner} wins` : 'Draw'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
      
      {/* Export */}
      {(benchmarkResults.length > 0 || tournamentResults.length > 0) && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex justify-center"
        >
          <Button
            onClick={handleExportResults}
            variant="outline"
            leftIcon={<Download className="w-4 h-4" />}
          >
            Export Results as JSON
          </Button>
        </motion.div>
      )}
    </div>
  )
}

export default ResearchPage
