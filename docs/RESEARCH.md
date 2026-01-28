# Research Documentation

## Cyber Warfare Strategy Game: A Platform for AI Agent Research

---

## 1. Introduction

This document provides comprehensive research documentation for the Cyber Warfare Strategy Game, designed to serve as a testbed for comparing classical game-theoretic AI (MinMax with Alpha-Beta pruning) against modern Deep Reinforcement Learning approaches (PPO with MCTS).

### 1.1 Research Motivation

Cyber security games present unique challenges for AI:
- **Imperfect Information**: Attackers don't know defender positions initially
- **Large State Spaces**: Network configurations create combinatorial explosion
- **Multi-objective Optimization**: Multiple victory conditions
- **Temporal Dynamics**: Actions have delayed consequences

### 1.2 Research Objectives

1. **Primary**: Compare hand-coded MinMax algorithms with learned Deep RL policies
2. **Secondary**: Investigate transfer learning in cyber security domains
3. **Tertiary**: Develop interpretable AI for security applications

---

## 2. Game Environment Design

### 2.1 State Space Analysis

The game state consists of:

| Component | Size | Description |
|-----------|------|-------------|
| Node States | O(n) | Compromised, patched, isolated status |
| Edge States | O(e) | Connection status (active/disabled) |
| Player Resources | O(1) | Action points, intel, compute |
| Vulnerability Map | O(n×v) | Vulnerabilities per node |
| Action History | O(t) | Past t turns of actions |

**Total State Space**: ~10^50 for medium-sized networks (20-30 nodes)

### 2.2 Action Space Analysis

| Player | Actions | Branching Factor |
|--------|---------|------------------|
| Attacker | 5 types | ~15-25 per turn |
| Defender | 5 types | ~20-40 per turn |

**Effective Branching Factor**: ~20-30 combined

### 2.3 Game Complexity Classification

| Metric | Value | Comparison |
|--------|-------|------------|
| State Space | ~10^50 | Between Checkers (10^21) and Chess (10^44) |
| Game Tree | ~10^80 | Between Chess (10^123) and Connect4 (10^21) |
| Average Length | 30-50 turns | Similar to Chess |

---

## 3. AI Agent Implementations

### 3.1 MinMax Agent Architecture

#### 3.1.1 Algorithm Components

```
MinMax Agent
├── Alpha-Beta Pruning
│   ├── Fail-soft implementation
│   └── Deep cutoffs with aspiration windows
├── Transposition Tables
│   ├── Zobrist hashing (64-bit keys)
│   ├── Two-tier table (depth-preferred + always-replace)
│   └── Entry: {hash, depth, score, flag, best_move}
├── Move Ordering
│   ├── Hash move (from TT)
│   ├── Killer moves (2 per ply)
│   ├── History heuristic (butterfly board)
│   └── Static ordering (captures first)
├── Iterative Deepening
│   ├── Time management with soft/hard limits
│   └── Principal variation tracking
└── Quiescence Search
    ├── Capture-only search
    └── Stand-pat evaluation
```

#### 3.1.2 Evaluation Function

The hand-crafted evaluation function uses weighted features:

**Material Features** (40% weight):
- Node control differential
- Critical node control
- Data exfiltration progress

**Positional Features** (30% weight):
- Network centrality of controlled nodes
- Adjacency to critical targets
- Isolation potential

**Tactical Features** (20% weight):
- Vulnerability exposure
- Detection probability
- Action efficiency

**Strategic Features** (10% weight):
- Network segment control
- Path diversity to objectives
- Resource advantage

#### 3.1.3 Performance Metrics

| Metric | Formula |
|--------|---------|
| Effective Branching Factor | EBF = N^(1/d) |
| Search Efficiency | SE = nodes_evaluated / total_nodes |
| Cutoff Rate | CR = beta_cutoffs / total_cutoffs |
| TT Hit Rate | HR = tt_hits / tt_probes |

### 3.2 Deep RL Agent Architecture

#### 3.2.1 Neural Network Design

```
Network Architecture (NumPy Implementation)
├── Input Layer
│   ├── Node features (n × 12)
│   ├── Edge features (e × 4)
│   ├── Global features (1 × 8)
│   └── Action mask (|A|)
├── Feature Extraction
│   ├── Dense Layer 1: 256 units, ReLU
│   ├── Dense Layer 2: 256 units, ReLU
│   └── Dense Layer 3: 128 units, ReLU
├── Policy Head
│   ├── Dense: 128 → 64, ReLU
│   └── Dense: 64 → |A|, Softmax
└── Value Head
    ├── Dense: 128 → 64, ReLU
    └── Dense: 64 → 1, Tanh
```

#### 3.2.2 PPO Algorithm Details

**Hyperparameters**:
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | 3e-4 | Standard PPO default |
| γ (discount) | 0.99 | Long-term planning |
| λ (GAE) | 0.95 | Bias-variance tradeoff |
| ε (clip) | 0.2 | Stable updates |
| Entropy coef | 0.01 | Exploration |
| Value coef | 0.5 | Balanced learning |
| Batch size | 64 | Memory efficient |
| Epochs | 4 | Per update |

**Loss Function**:
```
L(θ) = L_policy(θ) + c1 * L_value(θ) - c2 * H[π_θ]

L_policy = min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)
L_value = (V_θ(s) - V_target)²
H = -Σ π(a|s) log π(a|s)
```

#### 3.2.3 MCTS Integration

**UCB1 Selection**:
```
UCB(s,a) = Q(s,a) + c * P(s,a) * √(N(s)) / (1 + N(s,a))

where:
- Q(s,a): Average action value
- P(s,a): Prior probability from policy network
- N(s): Parent visit count
- N(s,a): Action visit count
- c: Exploration constant (√2)
```

**Search Parameters**:
| Parameter | Value |
|-----------|-------|
| Simulations | 100-800 |
| Exploration constant | √2 |
| Dirichlet noise α | 0.3 |
| Noise fraction | 0.25 |
| Temperature (start) | 1.0 |
| Temperature (end) | 0.1 |

---

## 4. Experimental Design

### 4.1 Experiment Types

#### E1: Algorithm Comparison
**Objective**: Compare MinMax vs Deep RL across difficulty levels

**Variables**:
- Independent: AI type, network topology, difficulty
- Dependent: Win rate, average game length, decision time
- Controlled: Random seeds, time limits

**Protocol**:
1. Generate 100 random networks per topology type
2. Play 10 games per network configuration
3. Alternate starting positions
4. Record all metrics

#### E2: Scalability Analysis
**Objective**: Measure performance degradation with network size

**Variables**:
- Independent: Network size (10, 20, 30, 50, 100 nodes)
- Dependent: Decision time, memory usage, move quality

**Protocol**:
1. Fixed topology pattern (scale up)
2. Measure response time at each scale
3. Profile memory usage
4. Expert evaluation of move quality

#### E3: Ablation Study
**Objective**: Measure contribution of each MinMax optimization

**Variants**:
- Base: Plain MinMax
- +AB: Alpha-Beta pruning
- +TT: Transposition tables
- +MO: Move ordering
- +QS: Quiescence search
- Full: All optimizations

### 4.2 Metrics and Analysis

#### Performance Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Win Rate | wins / total_games | Higher is better |
| Avg Game Length | mean(turns) | N/A (descriptive) |
| Decision Time | mean(move_time_ms) | Lower is better |
| Nodes Evaluated | nodes / move | Lower is better |
| ELO Rating | Relative skill | Higher is better |

#### Statistical Analysis
- **Significance Testing**: Two-tailed t-test (α=0.05)
- **Effect Size**: Cohen's d
- **Confidence Intervals**: 95% bootstrap CI
- **Multiple Comparisons**: Bonferroni correction

### 4.3 Data Collection

```json
{
  "game_id": "uuid",
  "timestamp": "ISO8601",
  "config": {
    "topology": "corporate",
    "difficulty": "hard",
    "attacker_ai": "minmax",
    "defender_ai": "deep_rl"
  },
  "result": {
    "winner": "attacker",
    "turns": 34,
    "score": [127, 89]
  },
  "moves": [
    {
      "turn": 1,
      "player": "attacker",
      "action": {"type": "SCAN", "target": 5},
      "time_ms": 234,
      "nodes_evaluated": 15420,
      "state_hash": "abc123"
    }
  ]
}
```

---

## 5. Expected Results and Analysis

### 5.1 Hypotheses

**H1**: MinMax will outperform Deep RL in small networks (n<20) due to exact search
**H2**: Deep RL will scale better to large networks (n>50)
**H3**: Hybrid approaches will show best overall performance
**H4**: MinMax will exhibit more consistent play (lower variance)

### 5.2 Analysis Framework

```
Results Analysis
├── Win Rate Comparison
│   ├── By topology type
│   ├── By difficulty level
│   └── By network size
├── Efficiency Analysis
│   ├── Time per decision
│   ├── Nodes per decision (MinMax)
│   └── Simulations per decision (MCTS)
├── Quality Analysis
│   ├── Expert evaluation
│   └── Strategic coherence
└── Scalability Analysis
    ├── Time complexity
    └── Space complexity
```

---

## 6. Publication Strategy

### 6.1 Target Venues

| Venue | Focus | Deadline | Notes |
|-------|-------|----------|-------|
| NeurIPS | Novel learning algorithms | May | Game-theoretic Deep RL |
| AAAI | Broad AI applications | Aug | AI for security |
| IJCAI | Multi-agent systems | Jan | Strategic reasoning |
| CoG | Game AI specifically | Mar | Applied game AI |
| AAMAS | Autonomous agents | Feb | Adversarial agents |

### 6.2 Paper Outline

1. **Introduction**
   - Motivation: AI for cyber security
   - Problem: Comparing classical vs learned agents
   - Contributions: Framework, implementations, analysis

2. **Related Work**
   - Game-theoretic cyber security
   - MinMax improvements
   - Deep RL in games
   - Hybrid approaches

3. **Problem Formulation**
   - Game environment model
   - State/action spaces
   - Reward design

4. **Methodology**
   - MinMax implementation details
   - Deep RL architecture
   - Experimental design

5. **Results**
   - Performance comparison
   - Scalability analysis
   - Ablation studies

6. **Discussion**
   - Interpretability
   - Transferability
   - Limitations

7. **Conclusion**
   - Key findings
   - Future work

### 6.3 Code and Data Availability

- **GitHub**: https://github.com/MMI122/CyberWarfareStrategyGame
- **Data**: Zenodo DOI (to be assigned)
- **Models**: HuggingFace Hub (to be uploaded)

---

## 7. Future Research Directions

### 7.1 Short-term (3-6 months)
- [ ] Complete ablation studies
- [ ] Benchmark on standard game AI metrics
- [ ] Develop interpretability tools
- [ ] Write first paper draft

### 7.2 Medium-term (6-12 months)
- [ ] Transfer learning experiments
- [ ] Real network topology integration
- [ ] Multi-agent scenarios
- [ ] Conference submission

### 7.3 Long-term (12+ months)
- [ ] Human-AI collaboration studies
- [ ] Online learning variants
- [ ] Security tool integration
- [ ] PhD thesis completion

---

## 8. Reproducibility

### 8.1 Environment Setup

```bash
# Python version
python 3.11+

# Create reproducible environment
pip install -r requirements.txt

# Verify installation
python -c "from backend.app.core import GameEngine; print('OK')"
```

### 8.2 Random Seeds

All experiments use controlled random seeds:
```python
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```

### 8.3 Hardware Specifications

Experiments run on:
- CPU: Intel i7/AMD Ryzen equivalent
- RAM: 16GB minimum
- GPU: Optional (NVIDIA CUDA 11+)

---

## 9. References

1. Knuth, D.E., Moore, R.W. (1975). "An analysis of alpha-beta pruning"
2. Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms"
3. Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play"
4. Boddy, M.S. et al. (2005). "Course of Action Generation for Cyber Security"
5. Alpcan, T., Başar, T. (2010). "Network Security: A Decision and Game-Theoretic Approach"

---

## Appendix A: Code Examples

### A.1 Running Experiments

```python
from backend.app.core.game_engine import GameEngine
from backend.app.ai.minmax_agent import MinMaxAgent
from backend.app.ai.deep_rl_agent import DeepRLAgent

# Create game
engine = GameEngine()
game = engine.create_game("corporate", "hard")

# Create agents
minmax = MinMaxAgent("attacker", depth=6)
deeprl = DeepRLAgent("defender", simulations=200)

# Play and collect data
results = []
while not game.is_game_over():
    start_time = time.time()
    
    if game.current_player == "attacker":
        action = minmax.get_action(game.state)
        nodes = minmax.last_nodes_evaluated
    else:
        action = deeprl.get_action(game.state)
        nodes = deeprl.last_simulations
    
    decision_time = time.time() - start_time
    game.execute_action(action)
    
    results.append({
        "turn": game.turn,
        "action": action.to_dict(),
        "time": decision_time,
        "nodes": nodes
    })
```

### A.2 Custom Evaluation Function

```python
def custom_evaluation(state):
    """
    Hand-crafted evaluation function for cyber warfare game.
    Returns score from attacker's perspective.
    """
    score = 0
    
    # Material: node control
    attacker_nodes = len(state.compromised_nodes)
    defender_nodes = len(state.patched_nodes)
    score += (attacker_nodes - defender_nodes) * 10
    
    # Position: centrality
    for node_id in state.compromised_nodes:
        score += state.network.centrality[node_id] * 5
    
    # Tactics: distance to critical
    min_distance = min_path_to_critical(state)
    score -= min_distance * 3
    
    # Strategy: data exfiltrated
    score += state.attacker_score * 2
    
    return score
```

---

*Document Version: 1.0 | Last Updated: 2024*
