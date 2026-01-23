#!/usr/bin/env python3
"""
Analysis and Comparison Functions for TCP Network Simulator

Provides functions for comparing TCP performance across different parameters
and generating analysis plots.
"""

from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

from simulator import TCPSimulator


def compare_loss_rates(loss_rates: List[float] = None,
                       bandwidth_mbps: float = 10.0,
                       rtt_ms: float = 40.0,
                       simulation_time: float = 30.0,
                       random_seed: int = None,
                       plot: bool = True,
                       verbose: bool = True) -> List[Dict]:
    """
    Compare TCP performance under different wireless loss rates.
    
    Args:
        loss_rates: List of loss rates to test (0.0 to 1.0)
        bandwidth_mbps: Link bandwidth
        rtt_ms: Base round-trip time
        simulation_time: Duration for each test
        random_seed: Random seed for reproducibility
        plot: Whether to generate comparison plot
        verbose: Whether to print progress
        
    Returns:
        List of result dictionaries
    """
    if loss_rates is None:
        loss_rates = [0.0, 0.01, 0.02, 0.05, 0.10]
    
    if verbose:
        print("\n" + "=" * 60)
        print(" Comparing TCP Performance vs Wireless Loss Rate")
        print("=" * 60)
    
    results = []
    
    for i, loss_rate in enumerate(loss_rates):
        if verbose:
            print(f"\n--- Testing {loss_rate*100:.1f}% loss rate ---")
        
        # Use incrementing seed for reproducibility but different runs
        seed = None if random_seed is None else random_seed + i
        
        sim = TCPSimulator(
            bandwidth_mbps=bandwidth_mbps,
            rtt_ms=rtt_ms,
            loss_rate=loss_rate,
            simulation_time=simulation_time,
            random_seed=seed
        )
        result = sim.run(verbose=verbose)
        result['configured_loss_rate'] = loss_rate
        results.append(result)
    
    if plot and results:
        # Ensure plots directory exists
        import os
        plots_dir = './plots'
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"loss_comparison_{bandwidth_mbps}mbps.png")
        plot_loss_comparison(results, plot_path)
    
    return results


def plot_loss_comparison(results: List[Dict], filename: str):
    """Generate comparison plot for loss rate experiments"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('TCP Performance vs Wireless Loss Rate', fontsize=14, fontweight='bold')
    
    loss_rates = [r['configured_loss_rate'] * 100 for r in results]
    
    # Throughput
    ax = axes[0]
    throughputs = [r['throughput_mbps'] for r in results]
    ax.plot(loss_rates, throughputs, marker='o', linewidth=2, markersize=10, color='#2ecc71')
    ax.fill_between(loss_rates, throughputs, alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Wireless Loss Rate (%)', fontsize=12)
    ax.set_ylabel('Throughput (Mbps)', fontsize=12)
    ax.set_title('Throughput Degradation')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Efficiency
    ax = axes[1]
    efficiencies = [r['efficiency_percent'] for r in results]
    ax.plot(loss_rates, efficiencies, marker='s', linewidth=2, markersize=10, color='#9b59b6')
    ax.fill_between(loss_rates, efficiencies, alpha=0.3, color='#9b59b6')
    ax.set_xlabel('Wireless Loss Rate (%)', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Link Utilization')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # RTT
    ax = axes[2]
    rtts = [r['avg_rtt_ms'] for r in results]
    ax.plot(loss_rates, rtts, marker='^', linewidth=2, markersize=10, color='#e74c3c')
    ax.set_xlabel('Wireless Loss Rate (%)', fontsize=12)
    ax.set_ylabel('Average RTT (ms)', fontsize=12)
    ax.set_title('Latency Impact')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {filename}")
    plt.close()


def print_comparison_summary(results: List[Dict]):
    """Print a summary table of comparison results"""
    print("\n" + "=" * 70)
    print(" COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Loss %':>8} {'Throughput':>12} {'Efficiency':>12} {'Avg RTT':>10} {'Timeouts':>10}")
    print("-" * 70)
    
    baseline_throughput = results[0]['throughput_mbps'] if results else 1
    
    for r in results:
        loss_pct = r['configured_loss_rate'] * 100
        throughput = r['throughput_mbps']
        efficiency = r['efficiency_percent']
        rtt = r['avg_rtt_ms']
        timeouts = r['timeouts']
        
        # Calculate degradation from baseline
        if baseline_throughput > 0:
            degradation = (1 - throughput / baseline_throughput) * 100
            deg_str = f" (-{degradation:.0f}%)" if loss_pct > 0 else ""
        else:
            deg_str = ""
        
        print(f"{loss_pct:7.1f}% {throughput:10.2f} Mbps{deg_str:>8} "
              f"{efficiency:10.1f}% {rtt:8.1f} ms {timeouts:10d}")
    
    print("=" * 70)
