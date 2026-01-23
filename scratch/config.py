#!/usr/bin/env python3
"""
Configuration Classes for TCP Network Simulator

Defines all configuration dataclasses for network, TCP, wireless, and simulation parameters.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class NetworkConfig:
    """Network link configuration parameters"""
    bandwidth_mbps: float = 10.0       # Link bandwidth in Mbps
    base_rtt_ms: float = 40.0          # Base round-trip time in ms
    queue_size_packets: int = 100      # Router queue capacity in packets
    mtu_bytes: int = 1500              # Maximum transmission unit in bytes
    
    @property
    def bandwidth_bytes_per_sec(self) -> float:
        """Convert bandwidth to bytes per second"""
        return self.bandwidth_mbps * 1e6 / 8
    
    @property
    def base_rtt_sec(self) -> float:
        """Convert RTT to seconds"""
        return self.base_rtt_ms / 1000.0


@dataclass 
class TCPConfig:
    """TCP protocol configuration parameters"""
    initial_cwnd: float = 10.0         # Initial congestion window (packets)
    initial_ssthresh: float = 64.0     # Initial slow start threshold
    min_cwnd: float = 1.0              # Minimum congestion window
    min_ssthresh: float = 2.0          # Minimum slow start threshold
    
    # RTT estimation (RFC 6298)
    rtt_alpha: float = 0.125           # SRTT smoothing factor
    rtt_beta: float = 0.25             # RTTVAR smoothing factor
    rto_k: float = 4.0                 # RTO = SRTT + K * RTTVAR
    min_rto_sec: float = 0.2           # Minimum RTO (200ms per RFC 6298)
    max_rto_sec: float = 60.0          # Maximum RTO (60s per RFC 6298)
    
    # Fast retransmit/recovery
    dup_ack_threshold: int = 3         # Duplicate ACKs for fast retransmit


@dataclass
class WirelessConfig:
    """Wireless channel configuration using Gilbert-Elliott burst loss model.
    
    The Gilbert-Elliott model is a 2-state Markov chain:
    - GOOD state: Low loss probability (good channel conditions)
    - BAD state: High loss probability (interference, fading, etc.)
    
    This models realistic wireless behavior where losses occur in bursts
    due to fading, interference, or temporary obstructions.
    """
    # Overall average loss rate (for compatibility)
    loss_rate: float = 0.02            # Target average loss probability
    
    # Gilbert-Elliott model parameters
    use_burst_loss: bool = True        # Enable burst loss model (vs simple Bernoulli)
    
    # State transition probabilities
    p_good_to_bad: float = 0.02        # Probability of transitioning GOOD -> BAD
    p_bad_to_good: float = 0.3         # Probability of transitioning BAD -> GOOD
    
    # Per-state loss probabilities  
    loss_prob_good: float = 0.001      # Loss probability in GOOD state (rare)
    loss_prob_bad: float = 0.25        # Loss probability in BAD state (bursty)
    
    # Average burst length = 1 / p_bad_to_good ≈ 3.3 packets
    # Time in bad state fraction = p_good_to_bad / (p_good_to_bad + p_bad_to_good) ≈ 6.25%
    # Effective loss rate ≈ 0.9375 * 0.001 + 0.0625 * 0.25 ≈ 1.7%
    
    def __post_init__(self):
        if not 0.0 <= self.loss_rate <= 1.0:
            raise ValueError(f"loss_rate must be in [0, 1], got {self.loss_rate}")
        if not 0.0 <= self.p_good_to_bad <= 1.0:
            raise ValueError(f"p_good_to_bad must be in [0, 1], got {self.p_good_to_bad}")
        if not 0.0 <= self.p_bad_to_good <= 1.0:
            raise ValueError(f"p_bad_to_good must be in [0, 1], got {self.p_bad_to_good}")
    
    @property
    def avg_burst_length(self) -> float:
        """Average number of packets in a burst (while in BAD state)"""
        return 1.0 / self.p_bad_to_good if self.p_bad_to_good > 0 else float('inf')
    
    @property
    def effective_loss_rate(self) -> float:
        """Calculate the effective average loss rate from Gilbert-Elliott params"""
        if not self.use_burst_loss:
            return self.loss_rate
        # Steady-state probability of being in BAD state
        pi_bad = self.p_good_to_bad / (self.p_good_to_bad + self.p_bad_to_good)
        pi_good = 1.0 - pi_bad
        return pi_good * self.loss_prob_good + pi_bad * self.loss_prob_bad


@dataclass
class SimulationConfig:
    """Simulation execution parameters"""
    duration_sec: float = 60.0         # Total simulation time in seconds
    time_step_sec: float = 0.001       # Time step granularity (1ms)
    metrics_interval_sec: float = 0.1  # How often to record metrics
    progress_interval_sec: float = 10.0  # How often to print progress
    random_seed: Optional[int] = None  # Random seed for reproducibility
    
    def __post_init__(self):
        if self.duration_sec <= 0:
            raise ValueError(f"duration_sec must be positive, got {self.duration_sec}")
        if self.time_step_sec <= 0:
            raise ValueError(f"time_step_sec must be positive, got {self.time_step_sec}")


@dataclass
class SimulatorConfig:
    """Complete simulator configuration"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    tcp: TCPConfig = field(default_factory=TCPConfig)
    wireless: WirelessConfig = field(default_factory=WirelessConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SimulatorConfig':
        """Create config from dictionary"""
        return cls(
            network=NetworkConfig(**config_dict.get('network', {})),
            tcp=TCPConfig(**config_dict.get('tcp', {})),
            wireless=WirelessConfig(**config_dict.get('wireless', {})),
            simulation=SimulationConfig(**config_dict.get('simulation', {}))
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'SimulatorConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)
    
    def save_json(self, filepath: str):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
