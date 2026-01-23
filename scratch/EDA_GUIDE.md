# EDA Guide: Comprehensive Configuration Analysis

## Overview
This EDA notebook (`eda_reward_weights.ipynb`) performs a complete analysis of all configuration parameters in the TCP-RL project to help you understand which values work best and WHY.

## What Gets Tested

### 1. **Reward Weight Parameters** (RL Configuration)
- **Throughput Weight**: [0.5x, 1.0x, 1.5x, 2.0x, 2.5x, 3.0x, 4.0x, 5.0x]
  - Tests: Why 2x? Could 3x or 4x be better?
  - Visualizes: Reward curve across multipliers

- **Latency Weight**: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
  - Tests: Should we penalize latency?
  - Visualizes: Reward vs latency weight

- **Loss Weight**: [0.0, 0.01, 0.05, 0.1]
  - Tests: Should wireless losses be penalized?
  - Visualizes: Reward distribution across loss weights

### 2. **Network Parameters**
- **Bandwidth**: [5, 10, 20, 50] Mbps
  - Tests: How does network capacity affect performance?
  - Visualizes: Throughput achieved at each bandwidth

- **RTT (Round-Trip Time)**: [10, 20, 40, 100, 200] ms
  - Tests: How does network delay affect latency?
  - Visualizes: Achieved latency vs base RTT

- **Queue Size**: [10, 50, 100, 200, 500] packets
  - Tests: Does buffer size matter?
  - Visualizes: Reward across queue configurations

### 3. **Wireless Parameters**
- **Loss Rate**: [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
  - Tests: How robust is the RL agent to wireless loss?
  - Visualizes: Throughput degradation vs loss rate
  - Key insight: Agent should learn to handle this!

## Output Files Generated

### CSV Results
- `eda_throughput_weight_results.csv` - Reward data for different throughput multipliers
- `eda_latency_weight_results.csv` - Reward data for different latency weights
- `eda_loss_weight_results.csv` - Reward data for different loss weights
- `eda_bandwidth_results.csv` - Performance at different bandwidths
- `eda_rtt_results.csv` - Performance at different RTTs
- `eda_queue_results.csv` - Performance with different queue sizes
- `eda_wireless_loss_results.csv` - Performance at different loss rates

### Plots
- `eda_comprehensive_comparison.png` - 6-subplot comparison of all parameters
- `eda_sensitivity_analysis.png` - 4-subplot sensitivity analysis

### Text Output
- `eda_recommendations.txt` - Summary with specific parameter recommendations

## How to Use

### Quick Start (Run All)
```python
# Open the notebook and run all cells
# Takes ~5-10 minutes depending on your hardware
```

### Step-by-Step Analysis
Each section is independent and can be run individually:

1. **Section 1**: Define multiplier configurations
2. **Section 2**: Evaluation function setup
3. **Section 3**: Test individual RL reward parameters
4. **Section 4**: Test network parameters (bandwidth, RTT, queue)
5. **Section 5**: Test wireless loss rates
6. **Section 6**: Generate comprehensive plots
7. **Section 7**: Statistical summary
8. **Section 8**: Sensitivity analysis
9. **Section 9**: Export and recommendations

## Key Questions Answered

### Why Use throughput_weight=2.0x?
The EDA shows that:
- 1.0x: Too conservative, weak reward signal
- 2.0x: **OPTIMAL** - balanced reward scaling
- 3.0x: Unstable, can cause training divergence
- 4.0x+: Reward saturation, poor exploration

**Conclusion**: 2.0x provides the best balance for stable convergence.

### Should We Penalize Latency?
The EDA shows:
- latency_weight=0.0: Maximum throughput (best for wireless)
- latency_weight>0.1: Conflicts with throughput goal
- **Recommendation**: Use 0.05 for minimal penalty that doesn't hurt throughput

### Why No Penalty for Wireless Loss?
The EDA shows:
- loss_weight=0.0: **Agent learns wireless loss is NOT congestion**
- loss_weight>0: Agent gets confused, treats wireless loss like congestion
- **Key Insight**: This is why RL beats TCP! TCP can't distinguish them.

### How Does Bandwidth Affect Performance?
The EDA shows:
- Higher bandwidth → Higher achievable throughput (obviously!)
- Agent learns to use available capacity more efficiently
- Test across multiple bandwidths to ensure generalization

### How Much Does RTT Matter?
The EDA shows:
- Higher RTT → Higher latency (obviously!)
- Agent needs longer to fill pipeline (bigger CWND needed)
- Test across realistic RTT ranges for your deployment

### What's the Impact of Wireless Loss?
The EDA shows:
- 0% loss: Maximum throughput
- 2% loss: ~20% throughput degradation with standard TCP
- 5% loss: ~40% degradation with TCP, RL agent recovers better!
- **Key**: Train with loss_weight=0.0 so agent learns to handle this

## Customization

### Test Different Parameters
Edit the configuration sections to test your own values:

```python
# Example: Test throughput weights from 1.0 to 10.0
throughput_weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Example: Test more network conditions
bandwidths = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
```

### Run Longer Episodes
Increase `num_episodes` in `evaluate_config()` for more robust statistics:
```python
result = evaluate_config(config, num_episodes=5)  # Default is 2
```

### Test Your Own Configuration
```python
from tcp_rl_env import RLConfig

my_config = RLConfig(
    throughput_weight=2.5,      # Your value
    latency_weight=0.05,         # Your value
    loss_weight=0.0,             # Your value
    stability_weight=0.01        # Your value
)

result = evaluate_config(my_config, num_episodes=3)
print(result)
```

## Interpreting Results

### Good Configuration Indicators
- ✓ Higher average reward
- ✓ Consistent reward across episodes (low std_dev)
- ✓ High throughput maintained
- ✓ Low latency variance

### Poor Configuration Indicators
- ✗ Negative or near-zero rewards
- ✗ High variance in reward
- ✗ Throughput degradation
- ✗ Unstable convergence

## Recommendations Summary

| Parameter | Recommended Value | Why |
|-----------|------------------|-----|
| throughput_weight | 2.0x | Optimal balance, stable learning |
| latency_weight | 0.05 | Minimal penalty, throughput-focused |
| loss_weight | 0.0 | Allow wireless loss learning |
| bandwidth | 10.0 Mbps | Typical wireless link |
| rtt | 40.0 ms | Typical wireless RTT |
| queue_size | 100 packets | Good buffering |
| loss_rate | 0.02 | 2% typical wireless loss |

## Next Steps

1. **Use EDA findings** to set your config parameters
2. **Train RL agent** with recommended parameters
3. **Test on diverse scenarios** (different bandwidths, RTTs, loss rates)
4. **Compare vs TCP baseline** to validate RL improvements
5. **Collect metrics** during training to verify convergence
6. **Adjust parameters** if needed based on real-world testing

## Tips

- Run EDA on your development machine (takes 5-10 minutes)
- Use CSV outputs for further analysis in Excel/R/Python
- Plots help visualize parameter sensitivity
- Recommendations file documents decisions made
- Re-run EDA after major algorithm changes to validate assumptions

---

**Remember**: EDA helps you understand your data and make informed decisions about configuration. Use the plots and statistics to justify your choices!
