#!/usr/bin/env python3
"""
Command-Line Interface for TCP Network Simulator

Provides argument parsing and main entry point for the simulator.
"""

import argparse
import json
from typing import Optional

from config import SimulatorConfig, NetworkConfig, WirelessConfig, SimulationConfig
from simulator import TCPSimulator
from analysis import compare_loss_rates, print_comparison_summary


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='TCP Network Simulator for Wireless Networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison across multiple loss rates')
    
    # Network parameters
    net_group = parser.add_argument_group('Network Parameters')
    net_group.add_argument('--bandwidth', type=float, default=10.0,
                           help='Link bandwidth in Mbps')
    net_group.add_argument('--rtt', type=float, default=40.0,
                           help='Base round-trip time in milliseconds')
    net_group.add_argument('--queue', type=int, default=100,
                           help='Queue size in packets')
    net_group.add_argument('--mtu', type=int, default=1500,
                           help='Maximum transmission unit in bytes')
    
    # Wireless parameters
    wireless_group = parser.add_argument_group('Wireless Parameters')
    wireless_group.add_argument('--loss', type=float, default=0.02,
                                help='Wireless packet loss rate (0.0 to 1.0)')
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument('--duration', type=float, default=60.0,
                           help='Simulation duration in seconds')
    sim_group.add_argument('--seed', type=int, default=None,
                           help='Random seed for reproducibility')
    
    # Output options
    out_group = parser.add_argument_group('Output Options')
    out_group.add_argument('--plot', action='store_true',
                           help='Generate result plots')
    out_group.add_argument('--output', type=str, default='plots/simulation_results.png',
                           help='Output filename for plots (saved in plots/ folder)')
    out_group.add_argument('--quiet', action='store_true',
                           help='Suppress progress output')
    out_group.add_argument('--json', type=str, default=None,
                           help='Save results to JSON file')
    out_group.add_argument('--config', type=str, default=None,
                           help='Load configuration from JSON file')
    
    # Comparison mode parameters
    compare_group = parser.add_argument_group('Comparison Mode Options')
    compare_group.add_argument('--loss-rates', type=float, nargs='+',
                               default=[0.0, 0.01, 0.02, 0.05, 0.10],
                               help='Loss rates to test in comparison mode')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load config from file if provided
    if args.config:
        config = SimulatorConfig.from_json(args.config)
    else:
        config = SimulatorConfig(
            network=NetworkConfig(
                bandwidth_mbps=args.bandwidth,
                base_rtt_ms=args.rtt,
                queue_size_packets=args.queue,
                mtu_bytes=args.mtu
            ),
            wireless=WirelessConfig(
                loss_rate=args.loss
            ),
            simulation=SimulationConfig(
                duration_sec=args.duration,
                random_seed=args.seed
            )
        )
    
    verbose = not args.quiet
    
    if args.compare:
        # Comparison mode: test multiple loss rates
        results = compare_loss_rates(
            loss_rates=args.loss_rates,
            bandwidth_mbps=args.bandwidth,
            rtt_ms=args.rtt,
            simulation_time=min(args.duration, 30.0),  # Shorter for comparison
            random_seed=args.seed,
            plot=args.plot,
            verbose=verbose
        )
        
        print_comparison_summary(results)
        
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.json}")
    else:
        # Single simulation mode
        sim = TCPSimulator(config=config)
        results = sim.run(verbose=verbose)
        
        if args.plot:
            sim.plot_results(filename=args.output, show=True)
        
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
