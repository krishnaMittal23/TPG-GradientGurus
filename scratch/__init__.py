#!/usr/bin/env python3
"""
TCP Network Simulator Package

A time-stepped discrete event simulator for TCP over wireless networks.
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

from analysis import (
    compare_loss_rates,
    plot_loss_comparison,
    print_comparison_summary
)

__version__ = "1.0.0"
__author__ = "RL-TCP Project"

__all__ = [
    'NetworkConfig',
    'TCPConfig',
    'WirelessConfig',
    'SimulationConfig',
    'SimulatorConfig',
    'Packet',
    'TCPSimulator',
    'compare_loss_rates',
    'plot_loss_comparison',
    'print_comparison_summary',
]
