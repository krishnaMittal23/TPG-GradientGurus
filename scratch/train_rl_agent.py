#!/usr/bin/env python3
"""
RL Training Script for TCP Congestion Control

Trains a PPO agent to optimize TCP congestion control in wireless networks.
The agent learns to distinguish between wireless loss and actual congestion.

Usage:
    python train_rl_agent.py                    # Train with defaults
    python train_rl_agent.py --timesteps 50000  # Train longer
    python train_rl_agent.py --eval             # Evaluate trained model
    python train_rl_agent.py --compare          # Compare RL vs TCP baseline

Author: RL-TCP Project
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

# RL imports
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Our environment
from tcp_rl_env import TCPWirelessEnv, RLConfig, make_env, evaluate_baseline_tcp
from python_tcp_simulator import NetworkConfig, WirelessConfig


# =============================================================================
# Custom Callback for Logging
# =============================================================================

class ThroughputCallback(BaseCallback):
    """Custom callback to log throughput during training"""
    
    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.throughputs = []
        self.timesteps = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation episode
            obs, info = self.eval_env.reset()
            episode_throughputs = []
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_throughputs.append(info['raw_state']['throughput_mbps'])
                done = terminated or truncated
            
            avg_throughput = np.mean(episode_throughputs)
            self.throughputs.append(avg_throughput)
            self.timesteps.append(self.num_timesteps)
            
            if self.verbose > 0:
                print(f"  Step {self.num_timesteps}: Avg Throughput = {avg_throughput:.2f} Mbps")
        
        return True


# =============================================================================
# Environment Factory
# =============================================================================

def make_training_env(loss_rate: float = 0.02, 
                      bandwidth_mbps: float = 10.0,
                      rtt_ms: float = 40.0,
                      rank: int = 0) -> TCPWirelessEnv:
    """Create a training environment with curriculum-like loss rate variation"""
    
    # Vary loss rate slightly for each parallel environment
    varied_loss = loss_rate * (0.8 + 0.4 * (rank / max(1, rank)))
    varied_loss = max(0.005, min(0.15, varied_loss))
    
    return TCPWirelessEnv(
        network_config=NetworkConfig(
            bandwidth_mbps=bandwidth_mbps,
            base_rtt_ms=rtt_ms,
        ),
        wireless_config=WirelessConfig(
            loss_rate=loss_rate  # Use base loss rate for consistency
        ),
        rl_config=RLConfig(
            steps_per_episode=100,
            step_duration_sec=0.5,
        )
    )


# =============================================================================
# Training Function
# =============================================================================

def train_agent(
    total_timesteps: int = 50000,
    loss_rate: float = 0.02,
    bandwidth_mbps: float = 10.0,
    rtt_ms: float = 40.0,
    algorithm: str = "PPO",
    save_path: str = "./models",
    log_path: str = "./logs",
    seed: int = 42,
    n_envs: int = 1,
    verbose: int = 1
) -> Tuple[object, Dict]:
    """
    Train an RL agent for TCP congestion control.
    
    Args:
        total_timesteps: Total training steps
        loss_rate: Wireless loss rate to train on
        bandwidth_mbps: Link bandwidth
        rtt_ms: Base RTT
        algorithm: RL algorithm (PPO, SAC, TD3)
        save_path: Where to save models
        log_path: Where to save tensorboard logs
        seed: Random seed
        n_envs: Number of parallel environments
        verbose: Verbosity level
        
    Returns:
        Tuple of (trained_model, training_info)
    """
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tcp_rl_{algorithm}_{timestamp}"
    
    print("=" * 60)
    print(" RL Training for TCP Congestion Control")
    print("=" * 60)
    print(f"Algorithm:      {algorithm}")
    print(f"Timesteps:      {total_timesteps:,}")
    print(f"Loss Rate:      {loss_rate * 100:.1f}%")
    print(f"Bandwidth:      {bandwidth_mbps} Mbps")
    print(f"RTT:            {rtt_ms} ms")
    print(f"Parallel Envs:  {n_envs}")
    print(f"Seed:           {seed}")
    print("=" * 60)
    
    # Create environment
    def make_env_fn():
        env = make_training_env(loss_rate, bandwidth_mbps, rtt_ms)
        env = Monitor(env)
        return env
    
    if n_envs > 1:
        env = DummyVecEnv([make_env_fn for _ in range(n_envs)])
    else:
        env = make_env_fn()
    
    # Create evaluation environment
    eval_env = make_training_env(loss_rate, bandwidth_mbps, rtt_ms)
    
    # Evaluate baseline first
    print("\n📊 Evaluating baseline TCP...")
    baseline = evaluate_baseline_tcp(eval_env, num_episodes=3)
    print(f"   Baseline throughput: {baseline['avg_throughput']:.2f} Mbps")
    print(f"   Baseline RTT: {baseline['avg_rtt']:.2f} ms")
    
    # Select algorithm
    if algorithm.upper() == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,           # Lower learning rate for stability
            n_steps=2048,                 # More steps per update
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,                 # HIGH entropy for exploration!
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Bigger network
                log_std_init=-0.5,        # Higher initial std for exploration
            ),
            verbose=verbose,
            tensorboard_log=log_path,
            seed=seed,
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            verbose=verbose,
            tensorboard_log=log_path,
            seed=seed,
        )
    elif algorithm.upper() == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            verbose=verbose,
            tensorboard_log=log_path,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, total_timesteps // 10),
        save_path=save_path,
        name_prefix=run_name
    )
    
    throughput_callback = ThroughputCallback(
        eval_env=eval_env,
        eval_freq=max(500, total_timesteps // 20),
        verbose=verbose
    )
    
    # Train!
    print("\n🚀 Starting training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, throughput_callback],
        progress_bar=True,
    )
    
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(save_path, f"{run_name}_final")
    model.save(final_model_path)
    print(f"\n✅ Model saved to: {final_model_path}")
    
    # Final evaluation
    print("\n📊 Final Evaluation...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    # Get final throughput
    final_throughputs = []
    for _ in range(5):
        obs, info = eval_env.reset()
        ep_throughputs = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_throughputs.append(info['raw_state']['throughput_mbps'])
            done = terminated or truncated
        final_throughputs.append(np.mean(ep_throughputs))
    
    final_throughput = np.mean(final_throughputs)
    improvement = ((final_throughput - baseline['avg_throughput']) / 
                   baseline['avg_throughput'] * 100)
    
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training Time:     {training_time:.1f} seconds")
    print(f"Mean Reward:       {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Baseline Thput:    {baseline['avg_throughput']:.2f} Mbps")
    print(f"RL Agent Thput:    {final_throughput:.2f} Mbps")
    print(f"Improvement:       {improvement:+.1f}%")
    print("=" * 60)
    
    # Plot training progress
    if throughput_callback.throughputs:
        plt.figure(figsize=(10, 5))
        plt.plot(throughput_callback.timesteps, throughput_callback.throughputs, 
                 'b-', linewidth=2, label='RL Agent')
        plt.axhline(y=baseline['avg_throughput'], color='r', linestyle='--', 
                    label=f'TCP Baseline ({baseline["avg_throughput"]:.2f} Mbps)')
        plt.xlabel('Training Steps')
        plt.ylabel('Throughput (Mbps)')
        plt.title('RL Agent Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, f'{run_name}_training.png'), dpi=150)
        print(f"📈 Training plot saved to: {save_path}/{run_name}_training.png")
        plt.close()
    
    training_info = {
        'baseline_throughput': baseline['avg_throughput'],
        'final_throughput': final_throughput,
        'improvement_percent': improvement,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'model_path': final_model_path,
    }
    
    return model, training_info


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_agent(
    model_path: str,
    loss_rate: float = 0.02,
    bandwidth_mbps: float = 10.0,
    rtt_ms: float = 40.0,
    n_episodes: int = 10,
    render: bool = False
) -> Dict:
    """
    Evaluate a trained agent.
    
    Args:
        model_path: Path to saved model
        loss_rate: Loss rate to evaluate on
        bandwidth_mbps: Link bandwidth
        rtt_ms: Base RTT
        n_episodes: Number of evaluation episodes
        render: Whether to print episode details
        
    Returns:
        Dictionary of evaluation results
    """
    print("=" * 60)
    print(" Evaluating Trained Agent")
    print("=" * 60)
    
    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Create environment
    env = make_env(loss_rate, bandwidth_mbps, rtt_ms)
    
    # Evaluate baseline
    print("\n📊 Baseline TCP Performance...")
    baseline = evaluate_baseline_tcp(env, num_episodes=n_episodes)
    
    # Evaluate RL agent
    print("\n🤖 RL Agent Performance...")
    rl_results = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_throughputs = []
        episode_rtts = []
        episode_rewards = []
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_throughputs.append(info['raw_state']['throughput_mbps'])
            episode_rtts.append(info['raw_state']['avg_rtt_ms'])
            episode_rewards.append(reward)
            
            done = terminated or truncated
        
        rl_results.append({
            'throughput': np.mean(episode_throughputs),
            'rtt': np.mean(episode_rtts),
            'reward': sum(episode_rewards)
        })
        
        if render:
            print(f"  Episode {ep+1}: Throughput={np.mean(episode_throughputs):.2f} Mbps, "
                  f"RTT={np.mean(episode_rtts):.2f} ms")
    
    rl_throughput = np.mean([r['throughput'] for r in rl_results])
    rl_rtt = np.mean([r['rtt'] for r in rl_results])
    rl_reward = np.mean([r['reward'] for r in rl_results])
    
    improvement = ((rl_throughput - baseline['avg_throughput']) / 
                   baseline['avg_throughput'] * 100)
    
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'TCP Baseline':>15} {'RL Agent':>15} {'Change':>12}")
    print("-" * 60)
    print(f"{'Throughput (Mbps)':<20} {baseline['avg_throughput']:>15.2f} {rl_throughput:>15.2f} {improvement:>+11.1f}%")
    print(f"{'RTT (ms)':<20} {baseline['avg_rtt']:>15.2f} {rl_rtt:>15.2f}")
    print(f"{'Reward':<20} {baseline['avg_reward']:>15.3f} {rl_reward:>15.3f}")
    print("=" * 60)
    
    return {
        'baseline': baseline,
        'rl_throughput': rl_throughput,
        'rl_rtt': rl_rtt,
        'rl_reward': rl_reward,
        'improvement': improvement
    }


# =============================================================================
# Comparison Function
# =============================================================================

def compare_across_loss_rates(
    model_path: str,
    loss_rates: List[float] = None,
    bandwidth_mbps: float = 10.0,
    rtt_ms: float = 40.0,
    n_episodes: int = 5
) -> Dict:
    """
    Compare RL agent vs TCP baseline across different loss rates.
    """
    if loss_rates is None:
        loss_rates = [0.01, 0.02, 0.03, 0.05, 0.10]
    
    print("=" * 60)
    print(" Comparing RL Agent vs TCP Across Loss Rates")
    print("=" * 60)
    
    model = PPO.load(model_path)
    results = []
    
    for loss_rate in loss_rates:
        print(f"\n--- Testing {loss_rate*100:.1f}% loss rate ---")
        
        env = make_env(loss_rate, bandwidth_mbps, rtt_ms)
        
        # Baseline
        baseline = evaluate_baseline_tcp(env, num_episodes=n_episodes)
        
        # RL agent
        rl_throughputs = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_throughputs = []
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                ep_throughputs.append(info['raw_state']['throughput_mbps'])
                done = terminated or truncated
            rl_throughputs.append(np.mean(ep_throughputs))
        
        rl_throughput = np.mean(rl_throughputs)
        improvement = ((rl_throughput - baseline['avg_throughput']) / 
                       max(0.01, baseline['avg_throughput']) * 100)
        
        results.append({
            'loss_rate': loss_rate,
            'tcp_throughput': baseline['avg_throughput'],
            'rl_throughput': rl_throughput,
            'improvement': improvement
        })
        
        print(f"   TCP: {baseline['avg_throughput']:.2f} Mbps, RL: {rl_throughput:.2f} Mbps ({improvement:+.1f}%)")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    loss_pcts = [r['loss_rate'] * 100 for r in results]
    plt.plot(loss_pcts, [r['tcp_throughput'] for r in results], 
             'r-o', linewidth=2, markersize=8, label='TCP Baseline')
    plt.plot(loss_pcts, [r['rl_throughput'] for r in results], 
             'b-s', linewidth=2, markersize=8, label='RL Agent')
    plt.xlabel('Wireless Loss Rate (%)')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(loss_pcts, [r['improvement'] for r in results], color='green', alpha=0.7)
    plt.xlabel('Wireless Loss Rate (%)')
    plt.ylabel('Improvement (%)')
    plt.title('RL Agent Improvement over TCP')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_vs_tcp_comparison.png', dpi=150)
    print(f"\n📈 Comparison plot saved to: rl_vs_tcp_comparison.png")
    plt.close()
    
    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate RL agent for TCP congestion control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument('--eval', type=str, default=None,
                        help='Evaluate a trained model (provide path)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Compare trained model across loss rates (provide path)')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=30000,
                        help='Total training timesteps')
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'SAC', 'TD3'],
                        help='RL algorithm to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Environment parameters
    parser.add_argument('--loss', type=float, default=0.02,
                        help='Wireless loss rate')
    parser.add_argument('--bandwidth', type=float, default=10.0,
                        help='Link bandwidth (Mbps)')
    parser.add_argument('--rtt', type=float, default=40.0,
                        help='Base RTT (ms)')
    
    # Output
    parser.add_argument('--save-path', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.eval:
        # Evaluation mode
        evaluate_agent(
            model_path=args.eval,
            loss_rate=args.loss,
            bandwidth_mbps=args.bandwidth,
            rtt_ms=args.rtt,
            render=True
        )
    elif args.compare:
        # Comparison mode
        compare_across_loss_rates(
            model_path=args.compare,
            bandwidth_mbps=args.bandwidth,
            rtt_ms=args.rtt
        )
    else:
        # Training mode
        model, info = train_agent(
            total_timesteps=args.timesteps,
            loss_rate=args.loss,
            bandwidth_mbps=args.bandwidth,
            rtt_ms=args.rtt,
            algorithm=args.algorithm,
            save_path=args.save_path,
            log_path=args.log_path,
            seed=args.seed,
            verbose=args.verbose
        )
        
        print(f"\n🎉 Training complete! Model saved to: {info['model_path']}")
        print(f"\nTo evaluate: python train_rl_agent.py --eval {info['model_path']}")
        print(f"To compare:  python train_rl_agent.py --compare {info['model_path']}")


if __name__ == "__main__":
    main()
