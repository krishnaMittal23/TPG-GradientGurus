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
    
    # Target values (for normalizing rewards)
    target_throughput_mbps: float = 8.0    # Achievable target (80% of 10 Mbps)
    target_rtt_ms: float = 150.0           # Lenient RTT target
    
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
        
    def _create_simulator(self) -> RLTCPSimulator:
        """Create a new simulator instance"""
        sim_config = SimulatorConfig(
            network=self.network_config,
            wireless=self.wireless_config,
            simulation=SimulationConfig(
                duration_sec=self.rl_config.steps_per_episode * self.rl_config.step_duration_sec,
                random_seed=None  # Random each episode
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
    
    def _calculate_reward(self, state: Dict) -> float:
        """
        Calculate reward based on current state.
        
        Goal: Maximize throughput while keeping latency reasonable.
        Key insight: We want the agent to be AGGRESSIVE with cwnd in wireless
        networks because random loss shouldn't trigger backoff.
        """
        rl = self.rl_config
        
        # Primary: Throughput reward (strongly weighted)
        # Use goodput = throughput * (1 - loss_rate) for "effective" throughput
        goodput_mbps = state['throughput_mbps'] * (1.0 - state['loss_rate'])
        throughput_ratio = goodput_mbps / rl.target_throughput_mbps
        
        # Logarithmic scaling to encourage pushing for high throughput
        # This gives higher reward for going from 5->10 Mbps than from 0->5 Mbps
        if throughput_ratio > 0:
            throughput_reward = rl.throughput_weight * (
                np.log1p(throughput_ratio * 5) / np.log1p(5)  # Normalized log scale
            )
        else:
            throughput_reward = -0.5  # Penalty for zero throughput
        
        # Secondary: Latency penalty (only penalize very high latency)
        # Allow up to 2x target RTT before any penalty kicks in
        latency_ratio = state['avg_rtt_ms'] / rl.target_rtt_ms
        latency_penalty = rl.latency_weight * max(0, (latency_ratio - 2.0) ** 2)
        
        # Tertiary: Congestion penalty (based on queue buildup, not loss)
        # Wireless loss is OK, but queue buildup means real congestion
        queue_occ = state['queue_occupancy'] / self.network_config.queue_size_packets
        congestion_penalty = 0.05 * max(0, queue_occ - 0.5) ** 2
        
        # Bonus for high utilization (>50% of bandwidth)
        utilization = state['throughput_mbps'] / self.network_config.bandwidth_mbps
        utilization_bonus = 0.1 * max(0, utilization - 0.5)
        
        reward = throughput_reward - latency_penalty - congestion_penalty + utilization_bonus
        
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
    results = []
    
    for _ in range(num_episodes):
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
        
        results.append({
            'throughput': np.mean(episode_throughput),
            'rtt': np.mean(episode_rtt),
            'total_reward': 0  # Not applicable for baseline
        })
    
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
