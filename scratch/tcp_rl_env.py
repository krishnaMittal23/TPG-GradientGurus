#!/usr/bin/env python3
"""
Gym Environment for RL-based TCP Congestion Control

This module provides an OpenAI Gymnasium environment interface for training
reinforcement learning agents to optimize TCP congestion control in wireless networks.

The key insight: standard TCP treats ALL packet loss as congestion, but in wireless
networks much of the loss is due to random interference. An RL agent can learn to
distinguish between these two types of loss and maintain higher throughput.

Author: RL-TCP Project
"""


import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import random

# Import our TCP simulator
from python_tcp_simulator import (
    TCPSimulator, SimulatorConfig, NetworkConfig, 
    WirelessConfig, SimulationConfig, TCPConfig, Packet
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RLConfig:
    """Configuration for the RL environment"""
    # Reward weights - tuned for aggressive throughput maximization
    throughput_weight: float = 2.0      # Primary: maximize throughput (doubled)
    latency_weight: float = 0.05        # Small penalty for very high latency
    loss_weight: float = 0.0            # NO penalty for loss (key for wireless!)
    stability_weight: float = 0.0       # Disabled - we want exploration
    
    # REMOVED: Fixed target_throughput_mbps - now uses capacity-relative metrics
    # Capacity estimation settings (for time-varying networks)
    capacity_ewma_alpha: float = 0.3    # EWMA smoothing for capacity estimation
    capacity_estimation_window: int = 10  # Steps to track for peak detection
    
    # RTT-based congestion detection
    target_rtt_multiplier: float = 1.5  # RTT > base_rtt * multiplier = congested
    min_rtt_ewma_alpha: float = 0.1     # Slow tracking of minimum RTT
    
    # Action space bounds - direct cwnd control
    # Using wide range to allow agent to explore
    min_cwnd: float = 10.0                 # Minimum cwnd (raised to prevent too-safe behavior)
    max_cwnd: float = 150.0                # Maximum cwnd (higher ceiling)
    
    # Episode configuration
    steps_per_episode: int = 100           # Shorter episodes for faster training
    step_duration_sec: float = 0.5         # Longer decision intervals
    
    # State normalization bounds
    max_rtt_ms: float = 500.0
    max_throughput_mbps: float = 15.0
    max_cwnd: float = 150.0
    max_in_flight: float = 150.0
    
    # Dynamic capacity simulation (for training robustness)
    enable_dynamic_capacity: bool = True   # Enable time-varying capacity
    capacity_variation_range: float = 0.3  # ±30% capacity variation
    capacity_change_interval: int = 20     # Steps between capacity changes


# =============================================================================
# RL-Enabled TCP Simulator
# =============================================================================

class RLTCPSimulator(TCPSimulator):
    """
    Extended TCP simulator that allows RL agent to control congestion window.
    
    This simulator runs for a fixed duration between RL decisions, then
    returns the observed state for the agent to make its next decision.
    """
    
    def __init__(self, config: SimulatorConfig = None, **kwargs):
        super().__init__(config, **kwargs)
        self.rl_override_cwnd = None  # If set, use this instead of TCP's cwnd
    
    def set_cwnd_override(self, cwnd: float):
        """Allow RL agent to override the congestion window"""
        self.rl_override_cwnd = max(1.0, cwnd)
    
    def clear_cwnd_override(self):
        """Return control to standard TCP"""
        self.rl_override_cwnd = None
    
    def step_simulation(self, duration_sec: float) -> Dict:
        """
        Run simulation for a specified duration and return state.
        
        Args:
            duration_sec: How long to run before returning
            
        Returns:
            Dictionary with current state observations
        """
        start_time = self.current_time
        start_bytes = self.bytes_received
        start_packets = self.packets_received
        rtt_samples = []
        
        end_time = start_time + duration_sec
        
        while self.current_time < end_time:
            # Apply RL cwnd override if set
            if self.rl_override_cwnd is not None:
                self.cwnd = self.rl_override_cwnd
            
            # Normal TCP operations
            self._send_packets()
            self._process_acks()
            self._check_timeouts()
            
            # Collect RTT samples
            for seq, pkt in list(self.in_flight.items()):
                if self.current_time >= pkt.arrival_time and pkt.arrival_time != float('inf'):
                    rtt_samples.append((self.current_time - pkt.send_time) * 1000)
            
            self.current_time += self._time_step
        
        # Calculate step metrics
        elapsed = max(0.001, self.current_time - start_time)
        bytes_delta = self.bytes_received - start_bytes
        throughput = bytes_delta * 8 / elapsed / 1e6
        
        avg_rtt = np.mean(rtt_samples) if rtt_samples else self.srtt * 1000
        rtt_var = np.std(rtt_samples) if len(rtt_samples) > 1 else 0.0
        
        packets_delta = self.packets_received - start_packets
        loss_rate = 1 - (packets_delta / max(1, self.packets_sent - start_packets + packets_delta))
        
        return {
            'throughput_mbps': throughput,
            'avg_rtt_ms': avg_rtt,
            'rtt_var_ms': rtt_var,
            'loss_rate': max(0, min(1, loss_rate)),
            'cwnd': self.cwnd,
            'ssthresh': self.ssthresh,
            'in_flight': len(self.in_flight),
            'queue_occupancy': self.queue_occupancy,
            'srtt_ms': self.srtt * 1000,
            'time': self.current_time,
            'bytes_received': self.bytes_received,
            'packets_received': self.packets_received,
        }
    
    def is_done(self) -> bool:
        """Check if simulation has completed"""
        return self.current_time >= self._duration


# =============================================================================
# Gym Environment
# =============================================================================


#STATE SPACE
#7 VARIABLES ON WHICH MODEL IS TRAINED
class TCPWirelessEnv(gym.Env):
    """
    OpenAI Gymnasium environment for RL-based TCP congestion control.
    
    State Space (7 dimensions):
        0: Normalized throughput (0 to 1)
        1: Normalized RTT (0 to 1)
        2: RTT variance indicator (0 to 1)
        3: Recent loss rate (0 to 1)
        4: Normalized congestion window (0 to 1)
        5: Normalized in-flight packets (0 to 1)
        6: Queue occupancy ratio (0 to 1)
        
    Action Space (continuous):
        Congestion window multiplier in [0.5, 2.0]
        
    Reward:
        Weighted combination of throughput, latency, and loss penalties
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self,
                 network_config: NetworkConfig = None,
                 wireless_config: WirelessConfig = None,
                 rl_config: RLConfig = None,
                 render_mode: str = None):
        """
        Initialize the TCP wireless environment.
        
        Args:
            network_config: Network parameters (bandwidth, RTT, queue)
            wireless_config: Wireless channel parameters (loss rate)
            rl_config: RL-specific configuration (rewards, bounds)
            render_mode: How to render ('human' for print, None for no render)
        """
        super().__init__()
        
        self.network_config = network_config or NetworkConfig()
        self.wireless_config = wireless_config or WirelessConfig()
        self.rl_config = rl_config or RLConfig()
        self.render_mode = render_mode
        
        # State space: 7 normalized features
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )
        
        # Action space: direct cwnd value (not multiplier)
        self.action_space = spaces.Box(
            low=np.array([self.rl_config.min_cwnd], dtype=np.float32),
            high=np.array([self.rl_config.max_cwnd], dtype=np.float32),
            dtype=np.float32
        )
        
        # Internal state
        self.simulator: Optional[RLTCPSimulator] = None
        self.current_step = 0
        self.episode_count = 0
        self.previous_state: Optional[Dict] = None
        self.episode_rewards: List[float] = []
        
        # Capacity estimation state (for capacity-agnostic rewards)
        self.estimated_capacity_mbps: float = 0.0  # EWMA of achievable throughput
        self.peak_throughput_history: List[float] = []  # Recent peaks for capacity detection
        self.min_rtt_estimate_ms: float = float('inf')  # Minimum observed RTT
        self.current_capacity_scale: float = 1.0  # For dynamic capacity simulation
        self.base_bandwidth_mbps: float = self.network_config.bandwidth_mbps
        
    def _create_simulator(self) -> RLTCPSimulator:
        """Create a new simulator instance with fresh random state"""
        # Generate a truly random seed for this episode
        import time
        random_seed = int(time.time() * 1000000) % (2**31) + np.random.randint(0, 1000000)
        
        sim_config = SimulatorConfig(
            network=self.network_config,
            wireless=self.wireless_config,
            simulation=SimulationConfig(
                duration_sec=self.rl_config.steps_per_episode * self.rl_config.step_duration_sec,
                random_seed=random_seed  # Fresh random seed each episode
            )
        )
        return RLTCPSimulator(config=sim_config)
    
    def _normalize_state(self, state: Dict) -> np.ndarray:
        """Convert raw state dict to normalized observation vector"""
        rl = self.rl_config
        
        obs = np.array([
            min(1.0, state['throughput_mbps'] / rl.max_throughput_mbps),
            min(1.0, state['avg_rtt_ms'] / rl.max_rtt_ms),
            min(1.0, state['rtt_var_ms'] / (rl.max_rtt_ms / 4)),  # Variance normalized
            state['loss_rate'],
            min(1.0, state['cwnd'] / rl.max_cwnd),
            min(1.0, state['in_flight'] / rl.max_in_flight),
            state['queue_occupancy'] / self.network_config.queue_size_packets,
        ], dtype=np.float32)
        
        return np.clip(obs, 0.0, 1.0)
    
    def _update_capacity_estimate(self, state: Dict):
        """
        Update running estimate of available network capacity.
        
        Uses multiple signals to estimate current capacity:
        1. EWMA of achieved throughput (smoothed)
        2. Peak throughput detection (recent maximum)
        3. Minimum RTT tracking (baseline latency)
        
        This allows capacity-relative rewards without knowing true capacity.
        """
        rl = self.rl_config
        throughput = state['throughput_mbps']
        rtt = state['avg_rtt_ms']
        
        # Track minimum RTT (approximates base propagation delay)
        if rtt > 0 and rtt < self.min_rtt_estimate_ms:
            # Slow adaptation to avoid noise
            self.min_rtt_estimate_ms = (
                (1 - rl.min_rtt_ewma_alpha) * self.min_rtt_estimate_ms + 
                rl.min_rtt_ewma_alpha * rtt
            ) if self.min_rtt_estimate_ms < float('inf') else rtt
        
        # Track peak throughput in sliding window
        self.peak_throughput_history.append(throughput)
        if len(self.peak_throughput_history) > rl.capacity_estimation_window:
            self.peak_throughput_history.pop(0)
        
        # Estimate capacity as recent peak (what we CAN achieve)
        recent_peak = max(self.peak_throughput_history) if self.peak_throughput_history else throughput
        
        # EWMA update of capacity estimate
        if self.estimated_capacity_mbps <= 0:
            self.estimated_capacity_mbps = throughput
        else:
            # Asymmetric update: faster increase, slower decrease
            # This helps track capacity increases quickly while being robust to temporary dips
            if throughput > self.estimated_capacity_mbps:
                alpha = rl.capacity_ewma_alpha * 1.5  # Faster upward tracking
            else:
                alpha = rl.capacity_ewma_alpha * 0.5  # Slower downward tracking
            
            self.estimated_capacity_mbps = (
                (1 - alpha) * self.estimated_capacity_mbps + 
                alpha * recent_peak
            )
    
    def _calculate_reward(self, state: Dict) -> float:
        """
        Calculate CAPACITY-AGNOSTIC reward based on current state.
        
        Key improvements over fixed-target reward:
        1. Uses estimated capacity instead of hardcoded target
        2. RTT-based congestion detection (works regardless of capacity)
        3. Utilization relative to estimated available bandwidth
        4. Robust to time-varying conditions (cross-traffic, fading, etc.)
        
        Goal: Maximize throughput relative to what's achievable, not a fixed target.
        """
        rl = self.rl_config
        
        # Update capacity estimate with current observation
        self._update_capacity_estimate(state)
        
        throughput = state['throughput_mbps']
        rtt = state['avg_rtt_ms']
        loss_rate = state['loss_rate']
        queue_occ = state['queue_occupancy'] / self.network_config.queue_size_packets
        
        # Goodput = throughput accounting for retransmissions
        goodput_mbps = throughput * (1.0 - loss_rate)
        
        # === CAPACITY-RELATIVE THROUGHPUT REWARD ===
        # Use estimated capacity instead of fixed target
        # If we don't have a good estimate yet, use conservative baseline
        effective_capacity = max(0.1, self.estimated_capacity_mbps)
        utilization = goodput_mbps / effective_capacity
        
        # Reward scales with utilization (0 to 1+)
        # Log scale rewards marginal gains while preventing runaway values
        if utilization > 0:
            throughput_reward = rl.throughput_weight * (
                np.log1p(utilization * 3) / np.log1p(3)  # Normalized to ~1.0 at full utilization
            )
        else:
            throughput_reward = -0.5  # Penalty for zero throughput
        
        # === RTT-BASED CONGESTION DETECTION ===
        # Compare current RTT to estimated minimum RTT
        # This works regardless of absolute capacity
        if self.min_rtt_estimate_ms > 0 and self.min_rtt_estimate_ms < float('inf'):
            rtt_inflation = rtt / self.min_rtt_estimate_ms
            # Penalize when RTT exceeds threshold (indicates queuing/congestion)
            if rtt_inflation > rl.target_rtt_multiplier:
                latency_penalty = rl.latency_weight * (rtt_inflation - rl.target_rtt_multiplier) ** 2
            else:
                latency_penalty = 0.0
        else:
            # Fallback: use absolute RTT threshold
            latency_penalty = rl.latency_weight * max(0, (rtt / 100.0 - 2.0) ** 2)
        
        # === QUEUE-BASED CONGESTION PENALTY ===
        # Penalize high queue occupancy (true congestion signal)
        congestion_penalty = 0.05 * max(0, queue_occ - 0.5) ** 2
        
        # === UTILIZATION IMPROVEMENT BONUS ===
        # Bonus for exceeding 70% of estimated capacity (encourages exploration)
        if utilization > 0.7:
            utilization_bonus = 0.15 * (utilization - 0.7)
        else:
            utilization_bonus = 0.0
        
        # === CAPACITY DISCOVERY BONUS ===
        # Small bonus for discovering higher capacity (encourages probing)
        if throughput > self.estimated_capacity_mbps * 0.95:
            discovery_bonus = 0.1
        else:
            discovery_bonus = 0.0
        
        reward = (
            throughput_reward 
            - latency_penalty 
            - congestion_penalty 
            + utilization_bonus 
            + discovery_bonus
        )
        
        return float(reward)
    
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (can include 'loss_rate' to vary difficulty)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Allow varying loss rate via options
        if options and 'loss_rate' in options:
            self.wireless_config.loss_rate = options['loss_rate']
        
        # Reset capacity estimation state
        self.estimated_capacity_mbps = 0.0
        self.peak_throughput_history = []
        self.min_rtt_estimate_ms = float('inf')
        
        # Simulate dynamic capacity (time-varying network conditions)
        if self.rl_config.enable_dynamic_capacity:
            # Random capacity scaling for this episode
            # Simulates: wireless rate adaptation, cross-traffic, fading, cellular scheduling
            variation = self.rl_config.capacity_variation_range
            self.current_capacity_scale = 1.0 + np.random.uniform(-variation, variation)
            self.network_config.bandwidth_mbps = self.base_bandwidth_mbps * self.current_capacity_scale
        else:
            self.current_capacity_scale = 1.0
            self.network_config.bandwidth_mbps = self.base_bandwidth_mbps
        
        # Create fresh simulator
        self.simulator = self._create_simulator()
        self.current_step = 0
        self.episode_count += 1
        self.previous_state = None
        self.episode_rewards = []
        
        # Run initial step to get starting state
        state = self.simulator.step_simulation(self.rl_config.step_duration_sec)
        obs = self._normalize_state(state)
        
        info = {
            'episode': self.episode_count,
            'raw_state': state,
            'loss_rate': self.wireless_config.loss_rate,
        }
        
        self.previous_state = state
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Congestion window multiplier
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.simulator is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Extract and clip action - now direct cwnd value
        new_cwnd = float(np.clip(
            action[0],
            self.rl_config.min_cwnd,
            self.rl_config.max_cwnd
        ))
        
        # Apply action: set cwnd directly
        self.simulator.set_cwnd_override(new_cwnd)
        
        # Simulate intra-episode capacity changes (time-varying network)
        if (self.rl_config.enable_dynamic_capacity and 
            self.current_step > 0 and 
            self.current_step % self.rl_config.capacity_change_interval == 0):
            # Gradual capacity shift (not abrupt, more realistic)
            variation = self.rl_config.capacity_variation_range * 0.5  # Smaller intra-episode changes
            delta = np.random.uniform(-variation, variation)
            self.current_capacity_scale = np.clip(
                self.current_capacity_scale + delta,
                1.0 - self.rl_config.capacity_variation_range,
                1.0 + self.rl_config.capacity_variation_range
            )
            self.network_config.bandwidth_mbps = self.base_bandwidth_mbps * self.current_capacity_scale
            # Update simulator's bandwidth (affects queuing model)
            self.simulator._bandwidth = self.network_config.bandwidth_mbps * 1e6 / 8
        
        # Run simulation step
        state = self.simulator.step_simulation(self.rl_config.step_duration_sec)
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(state)
        self.episode_rewards.append(reward)
        
        # Check termination
        terminated = self.simulator.is_done()
        truncated = self.current_step >= self.rl_config.steps_per_episode
        
        # Create observation
        obs = self._normalize_state(state)
        
        info = {
            'step': self.current_step,
            'raw_state': state,
            'action_cwnd': new_cwnd,
            'episode_reward_sum': sum(self.episode_rewards),
            # Capacity estimation debug info
            'estimated_capacity_mbps': self.estimated_capacity_mbps,
            'current_capacity_scale': self.current_capacity_scale,
            'min_rtt_estimate_ms': self.min_rtt_estimate_ms,
            'actual_bandwidth_mbps': self.network_config.bandwidth_mbps,
        }
        
        self.previous_state = state
        
        if self.render_mode == 'human':
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the current state"""
        if self.previous_state is None:
            return
        
        state = self.previous_state
        print(f"\n--- Step {self.current_step} ---")
        print(f"Throughput: {state['throughput_mbps']:.2f} Mbps")
        print(f"RTT: {state['avg_rtt_ms']:.2f} ms")
        print(f"Loss Rate: {state['loss_rate']*100:.1f}%")
        print(f"CWND: {state['cwnd']:.1f}")
        print(f"In-flight: {state['in_flight']}")
        if self.episode_rewards:
            print(f"Total Reward: {sum(self.episode_rewards):.3f}")
    
    def close(self):
        """Clean up resources"""
        self.simulator = None


# =============================================================================
# Training Utilities
# =============================================================================

def make_env(loss_rate: float = 0.02, 
             bandwidth_mbps: float = 10.0,
             rtt_ms: float = 40.0) -> TCPWirelessEnv:
    """
    Factory function to create a configured environment.
    
    Args:
        loss_rate: Wireless packet loss rate
        bandwidth_mbps: Link bandwidth
        rtt_ms: Base round-trip time
        
    Returns:
        Configured TCPWirelessEnv instance
    """
    return TCPWirelessEnv(
        network_config=NetworkConfig(
            bandwidth_mbps=bandwidth_mbps,
            base_rtt_ms=rtt_ms,
        ),
        wireless_config=WirelessConfig(
            loss_rate=loss_rate
        )
    )


def evaluate_baseline_tcp(env: TCPWirelessEnv, 
                          num_episodes: int = 5) -> Dict:
    """
    Evaluate standard TCP (no RL intervention) as a baseline.
    
    For baseline, we let TCP control its own cwnd by not overriding it.
    
    Args:
        env: The environment to evaluate in
        num_episodes: Number of episodes to average over
        
    Returns:
        Dictionary of average metrics
    """
    # Log the environment's loss rate
    print(f"   evaluate_baseline_tcp: env.wireless_config.loss_rate = {env.wireless_config.loss_rate}")
    
    results = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_throughput = []
        episode_rtt = []
        
        # Clear the cwnd override to let TCP control itself
        if env.simulator:
            env.simulator.clear_cwnd_override()
        
        while True:
            # Run simulation step without changing cwnd
            state = env.simulator.step_simulation(env.rl_config.step_duration_sec)
            env.current_step += 1
            
            episode_throughput.append(state['throughput_mbps'])
            episode_rtt.append(state['avg_rtt_ms'])
            
            if env.simulator.is_done() or env.current_step >= env.rl_config.steps_per_episode:
                break
        
        ep_avg = np.mean(episode_throughput)
        results.append({
            'throughput': ep_avg,
            'rtt': np.mean(episode_rtt),
            'total_reward': 0  # Not applicable for baseline
        })
        print(f"   Episode {ep+1}: Baseline throughput = {ep_avg:.2f} Mbps")
    
    avg_throughput = np.mean([r['throughput'] for r in results])
    print(f"   Average baseline throughput: {avg_throughput:.2f} Mbps")
    
    return {
        'avg_throughput': np.mean([r['throughput'] for r in results]),
        'avg_rtt': np.mean([r['rtt'] for r in results]),
        'avg_reward': np.mean([r['total_reward'] for r in results]),
        'std_throughput': np.std([r['throughput'] for r in results]),
    }


# =============================================================================
# Main - Demo and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" TCP Wireless RL Environment - Demo")
    print("=" * 60)
    
    # Create environment
    env = make_env(loss_rate=0.02, bandwidth_mbps=10.0, rtt_ms=40.0)
    
    print("\n1. Environment Info:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    print("\n2. Running baseline TCP evaluation...")
    baseline = evaluate_baseline_tcp(env, num_episodes=3)
    print(f"   Baseline throughput: {baseline['avg_throughput']:.2f} +/- {baseline['std_throughput']:.2f} Mbps")
    print(f"   Baseline RTT: {baseline['avg_rtt']:.2f} ms")
    print(f"   Baseline reward: {baseline['avg_reward']:.3f}")
    
    print("\n3. Testing random policy...")
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"   Step {step}: action={action[0]:.2f}, "
                  f"throughput={info['raw_state']['throughput_mbps']:.2f}, "
                  f"reward={reward:.3f}")
        
        if terminated or truncated:
            break
    
    print(f"   Random policy total reward: {total_reward:.3f}")
    
    env.close()
    print("\n" + "=" * 60)
    print(" Environment ready for RL training!")
    print("=" * 60)
