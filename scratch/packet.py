#!/usr/bin/env python3
"""
Packet Data Structure for TCP Simulator

Represents TCP packets in the simulation.
"""

from dataclasses import dataclass


@dataclass
class Packet:
    """Represents a TCP packet in the simulation"""
    seq_num: int
    size: int
    send_time: float
    arrival_time: float = 0.0
    is_retransmit: bool = False
