#!/usr/bin/env python3
"""
TCP Network Simulator for Wireless Networks

A time-stepped discrete event simulator that demonstrates TCP's weakness
in wireless networks where random packet loss is misinterpreted as congestion.

Author: RL-TCP Project
Purpose: Training environment for RL-based congestion control

NOTE: This module now uses a modular architecture. Import from:
  - config: NetworkConfig, TCPConfig, WirelessConfig, SimulationConfig, SimulatorConfig
  - simulator: TCPSimulator
  - packet: Packet
  - analysis: compare_loss_rates, print_comparison_summary, plot_loss_comparison
  - cli: parse_args, main

For backward compatibility, all classes and functions are re-exported here.
"""

from config import (
    NetworkConfig,
    TCPConfig,
    WirelessConfig,
    SimulationConfig,
    SimulatorConfig
)
from packet import Packet
from simulator import TCPSimulator
from analysis import compare_loss_rates, print_comparison_summary
from cli import parse_args, main

# Re-export for backward compatibility
__all__ = [
    'NetworkConfig',
    'TCPConfig',
    'WirelessConfig',
    'SimulationConfig',
    'SimulatorConfig',
    'Packet',
    'TCPSimulator',
    'compare_loss_rates',
    'print_comparison_summary',
    'parse_args',
    'main',
]


if __name__ == "__main__":
    main()
