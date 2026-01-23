#!/usr/bin/env python3
"""
TCP Network Simulator Core Module

Implements the main TCPSimulator class for time-stepped discrete event simulation
of TCP over wireless networks.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from config import SimulatorConfig
from packet import Packet


class TCPSimulator:
    """
    Time-stepped TCP simulator over a wireless link.
    
    Implements TCP NewReno congestion control to demonstrate how TCP
    misinterprets wireless packet loss as network congestion.
    
    Key behaviors:
    - Slow Start: cwnd increases by 1 MSS per ACK until ssthresh
    - Congestion Avoidance: cwnd increases by 1/cwnd per ACK after ssthresh
    - Fast Retransmit: 3 duplicate ACKs trigger retransmit + halve cwnd
    - Timeout: Reset cwnd to 1 MSS, ssthresh to cwnd/2
    """
    
    def __init__(self, config: SimulatorConfig = None, **kwargs):
        """
        Initialize TCP simulator.
        
        Args:
            config: SimulatorConfig object with all parameters
            **kwargs: Override individual parameters:
                - bandwidth_mbps, rtt_ms, loss_rate, queue_size,
                  simulation_time, mtu, random_seed
        """
        # Create default config if not provided
        if config is None:
            config = SimulatorConfig()
        
        # Apply any kwargs overrides
        if 'bandwidth_mbps' in kwargs:
            config.network.bandwidth_mbps = kwargs['bandwidth_mbps']
        if 'rtt_ms' in kwargs:
            config.network.base_rtt_ms = kwargs['rtt_ms']
        if 'loss_rate' in kwargs:
            config.wireless.loss_rate = kwargs['loss_rate']
        if 'queue_size' in kwargs:
            config.network.queue_size_packets = kwargs['queue_size']
        if 'simulation_time' in kwargs:
            config.simulation.duration_sec = kwargs['simulation_time']
        if 'mtu' in kwargs:
            config.network.mtu_bytes = kwargs['mtu']
        if 'random_seed' in kwargs:
            config.simulation.random_seed = kwargs['random_seed']
        
        self.config = config
        
        # Set random seed for reproducibility
        if config.simulation.random_seed is not None:
            random.seed(config.simulation.random_seed)
            np.random.seed(config.simulation.random_seed)
        
        # Extract frequently used values for cleaner code
        self._bandwidth = config.network.bandwidth_bytes_per_sec
        self._base_rtt = config.network.base_rtt_sec
        self._loss_rate = config.wireless.loss_rate
        self._queue_size = config.network.queue_size_packets
        self._mtu = config.network.mtu_bytes
        self._duration = config.simulation.duration_sec
        self._time_step = config.simulation.time_step_sec
        
        # Initialize state
        self._reset_state()
    
    def _reset_state(self):
        """Reset all simulation state to initial values"""
        tcp = self.config.tcp
        
        # TCP state
        self.cwnd = tcp.initial_cwnd
        self.ssthresh = tcp.initial_ssthresh
        self.srtt = self._base_rtt
        self.rttvar = self._base_rtt / 2  # Initial RTTVAR estimate
        
        # Simulation state
        self.current_time = 0.0
        
        # Packet tracking
        self.next_seq = 0
        self.last_acked = -1
        self.in_flight: Dict[int, Packet] = {}
        self.dup_ack_count = 0
        self.last_ack_seq = -1
        
        # Queue state
        self.queue_occupancy = 0
        
        # Counters
        self.bytes_sent = 0
        self.bytes_received = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_lost_wireless = 0
        self.packets_lost_queue = 0
        self.retransmissions = 0
        self.timeouts = 0
        
        # History for plotting
        self.throughput_history: List[Tuple[float, float]] = []
        self.cwnd_history: List[Tuple[float, float]] = []
        self.rtt_history: List[Tuple[float, float]] = []
        self.queue_history: List[Tuple[float, int]] = []
        self.loss_events: List[float] = []  # Timestamps of loss events
        
    def _get_queue_delay(self) -> float:
        """Calculate queuing delay based on queue occupancy"""
        packet_service_time = self._mtu / self._bandwidth
        return self.queue_occupancy * packet_service_time
    
    def _get_current_rtt(self) -> float:
        """Get RTT including queuing delay"""
        return self._base_rtt + self._get_queue_delay()
    
    def _calculate_rto(self) -> float:
        """Calculate retransmission timeout using RFC 6298 algorithm"""
        tcp = self.config.tcp
        rto = self.srtt + tcp.rto_k * self.rttvar
        return max(tcp.min_rto_sec, min(tcp.max_rto_sec, rto))
    
    def _send_packets(self):
        """Send packets up to congestion window limit"""
        tcp = self.config.tcp
        
        # Calculate how many new packets we can send
        can_send = int(self.cwnd) - len(self.in_flight)
        
        for _ in range(max(0, can_send)):
            # Check queue capacity (congestion at router)
            if self.queue_occupancy >= self._queue_size:
                self.packets_lost_queue += 1
                # Queue drop = true congestion signal
                self.ssthresh = max(self.cwnd / 2, tcp.min_ssthresh)
                self.cwnd = max(self.ssthresh, tcp.min_cwnd)
                self.loss_events.append(self.current_time)
                break  # Stop sending when queue is full
            
            # Create new packet
            packet = Packet(
                seq_num=self.next_seq,
                size=self._mtu,
                send_time=self.current_time
            )
            
            self.next_seq += 1
            self.packets_sent += 1
            self.bytes_sent += self._mtu
            
            # Simulate wireless channel - random loss
            if random.random() < self._loss_rate:
                self.packets_lost_wireless += 1
                # Packet is lost but TCP doesn't know yet!
                # It will only find out when timeout occurs
                packet.arrival_time = float('inf')  # Never arrives
                self.in_flight[packet.seq_num] = packet
                self.queue_occupancy = min(self._queue_size, self.queue_occupancy + 1)
                continue
            
            # Packet survives - calculate when ACK will arrive
            current_rtt = self._get_current_rtt()
            # Add small jitter for realism (±5% of RTT)
            jitter = current_rtt * 0.05 * (2 * random.random() - 1)
            packet.arrival_time = self.current_time + current_rtt + jitter
            
            self.in_flight[packet.seq_num] = packet
            self.queue_occupancy = min(self._queue_size, self.queue_occupancy + 1)
    
    def _process_acks(self):
        """Process arriving ACKs and update TCP state"""
        tcp = self.config.tcp
        acks_received = []
        
        # Find packets whose ACKs have arrived
        for seq, packet in list(self.in_flight.items()):
            if self.current_time >= packet.arrival_time:
                acks_received.append(seq)
        
        if not acks_received:
            return
        
        # Process ACKs in sequence order
        acks_received.sort()
        
        for seq in acks_received:
            packet = self.in_flight.pop(seq)
            
            # Update RTT estimate using RFC 6298 algorithm
            # Only use non-retransmitted packets for RTT estimation
            if not packet.is_retransmit:
                sample_rtt = self.current_time - packet.send_time
                
                if self.rtt_history:  # Not first sample
                    # RTTVAR = (1 - beta) * RTTVAR + beta * |SRTT - R'|
                    self.rttvar = (1 - tcp.rtt_beta) * self.rttvar + \
                                  tcp.rtt_beta * abs(self.srtt - sample_rtt)
                    # SRTT = (1 - alpha) * SRTT + alpha * R'
                    self.srtt = (1 - tcp.rtt_alpha) * self.srtt + \
                                tcp.rtt_alpha * sample_rtt
                else:
                    # First RTT measurement
                    self.srtt = sample_rtt
                    self.rttvar = sample_rtt / 2
                
                self.rtt_history.append((self.current_time, sample_rtt * 1000))
            
            # Update counters
            self.bytes_received += packet.size
            self.packets_received += 1
            self.queue_occupancy = max(0, self.queue_occupancy - 1)
            
            # Check for duplicate ACK (simplified)
            if seq <= self.last_ack_seq:
                self.dup_ack_count += 1
                if self.dup_ack_count >= tcp.dup_ack_threshold:
                    # Fast retransmit/recovery
                    self.ssthresh = max(self.cwnd / 2, tcp.min_ssthresh)
                    self.cwnd = self.ssthresh + tcp.dup_ack_threshold
                    self.dup_ack_count = 0
                    self.loss_events.append(self.current_time)
            else:
                # New ACK - reset dup count and grow window
                self.dup_ack_count = 0
                self.last_ack_seq = seq
                
                # TCP NewReno congestion control
                if self.cwnd < self.ssthresh:
                    # Slow start: exponential growth
                    self.cwnd += 1.0
                else:
                    # Congestion avoidance: linear growth
                    self.cwnd += 1.0 / self.cwnd
            
            self.last_acked = max(self.last_acked, seq)
    
    def _check_timeouts(self):
        """Check for packet timeouts and handle retransmission"""
        tcp = self.config.tcp
        rto = self._calculate_rto()
        
        timed_out = []
        for seq, packet in self.in_flight.items():
            time_since_send = self.current_time - packet.send_time
            if time_since_send > rto:
                timed_out.append(seq)
        
        if timed_out:
            self.timeouts += len(timed_out)
            
            # TIMEOUT - This is the key TCP behavior!
            # TCP interprets ALL loss as congestion, even wireless loss
            self.ssthresh = max(self.cwnd / 2, tcp.min_ssthresh)
            self.cwnd = tcp.min_cwnd  # Reset to 1 MSS (harsh penalty)
            
            self.loss_events.append(self.current_time)
            
            # Remove timed out packets and free queue space
            for seq in timed_out:
                del self.in_flight[seq]
                self.queue_occupancy = max(0, self.queue_occupancy - 1)
                self.retransmissions += 1
    
    def _record_metrics(self):
        """Record simulation metrics at current time"""
        self.throughput_history.append((self.current_time, self.bytes_received))
        self.cwnd_history.append((self.current_time, self.cwnd))
        self.queue_history.append((self.current_time, self.queue_occupancy))
    
    def run(self, verbose: bool = True) -> Dict:
        """Run the simulation
        
        Args:
            verbose: Whether to print progress and results
            
        Returns:
            Dictionary of simulation results
        """
        sim = self.config.simulation
        
        if verbose:
            self._print_header()
        
        last_report_time = 0.0
        last_metric_time = 0.0
        
        while self.current_time < self._duration:
            # TCP sender behavior each time step
            self._send_packets()
            self._process_acks()
            self._check_timeouts()
            
            # Record metrics periodically
            if self.current_time - last_metric_time >= sim.metrics_interval_sec:
                self._record_metrics()
                last_metric_time = self.current_time
            
            # Progress report
            if verbose and self.current_time - last_report_time >= sim.progress_interval_sec:
                self._print_progress()
                last_report_time = self.current_time
            
            # Advance simulation time
            self.current_time += self._time_step
        
        results = self.get_results()
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_header(self):
        """Print simulation configuration header"""
        net = self.config.network
        wireless = self.config.wireless
        sim = self.config.simulation
        
        print("=" * 60)
        print(" TCP Wireless Network Simulation")
        print("=" * 60)
        print(f"Bandwidth:          {net.bandwidth_mbps:.1f} Mbps")
        print(f"Base RTT:           {net.base_rtt_ms:.1f} ms")
        print(f"Wireless Loss Rate: {wireless.loss_rate * 100:.1f}%")
        print(f"Queue Size:         {net.queue_size_packets} packets")
        print(f"Simulation Time:    {sim.duration_sec:.1f} seconds")
        if sim.random_seed is not None:
            print(f"Random Seed:        {sim.random_seed}")
        print("=" * 60)
    
    def _print_progress(self):
        """Print progress during simulation"""
        elapsed = max(0.001, self.current_time)
        throughput = self.bytes_received * 8 / elapsed / 1e6
        print(f"  t={self.current_time:5.0f}s: RX={self.packets_received:6d} pkts, "
              f"Throughput={throughput:5.2f} Mbps, CWND={self.cwnd:5.1f}")
    
    def get_results(self) -> Dict:
        """Get comprehensive simulation results"""
        duration = max(0.001, self._duration)
        
        # Calculate throughput
        avg_throughput = self.bytes_received * 8 / duration / 1e6
        max_throughput = self.config.network.bandwidth_mbps
        efficiency = (avg_throughput / max_throughput * 100) if max_throughput > 0 else 0
        
        # Calculate RTT statistics
        if self.rtt_history:
            rtts = [r for _, r in self.rtt_history]
            avg_rtt = np.mean(rtts)
            min_rtt = np.min(rtts)
            max_rtt = np.max(rtts)
            rtt_std = np.std(rtts)
        else:
            avg_rtt = min_rtt = max_rtt = rtt_std = 0.0
        
        # Calculate loss rates
        total_lost = self.packets_lost_wireless + self.packets_lost_queue
        packets_sent = max(1, self.packets_sent)
        
        return {
            # Throughput metrics
            'throughput_mbps': avg_throughput,
            'max_throughput_mbps': max_throughput,
            'efficiency_percent': efficiency,
            
            # Latency metrics
            'avg_rtt_ms': avg_rtt,
            'min_rtt_ms': min_rtt,
            'max_rtt_ms': max_rtt,
            'rtt_std_ms': rtt_std,
            
            # Packet metrics
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'bytes_received': self.bytes_received,
            
            # Loss metrics
            'packets_lost_wireless': self.packets_lost_wireless,
            'packets_lost_queue': self.packets_lost_queue,
            'total_loss_rate': total_lost / packets_sent,
            'wireless_loss_rate': self.packets_lost_wireless / packets_sent,
            'queue_loss_rate': self.packets_lost_queue / packets_sent,
            
            # TCP behavior metrics
            'final_cwnd': self.cwnd,
            'final_ssthresh': self.ssthresh,
            'timeouts': self.timeouts,
            'retransmissions': self.retransmissions,
            'loss_events': len(self.loss_events),
            
            # Configuration (for reference)
            'config_loss_rate': self._loss_rate,
            'config_bandwidth_mbps': self.config.network.bandwidth_mbps,
            'config_rtt_ms': self.config.network.base_rtt_ms,
        }
    
    def _print_results(self, results: Dict):
        """Print formatted simulation results"""
        print(f"\n{'=' * 60}")
        print(" SIMULATION RESULTS")
        print(f"{'=' * 60}")
        
        print("\n📊 THROUGHPUT:")
        print(f"   Average:    {results['throughput_mbps']:.2f} Mbps")
        print(f"   Maximum:    {results['max_throughput_mbps']:.2f} Mbps")
        print(f"   Efficiency: {results['efficiency_percent']:.1f}%")
        
        print("\n⏱️  LATENCY:")
        print(f"   Average RTT: {results['avg_rtt_ms']:.2f} ms")
        print(f"   Min RTT:     {results['min_rtt_ms']:.2f} ms")
        print(f"   Max RTT:     {results['max_rtt_ms']:.2f} ms")
        
        print("\n📦 PACKETS:")
        print(f"   Sent:     {results['packets_sent']:,}")
        print(f"   Received: {results['packets_received']:,}")
        print(f"   Bytes RX: {results['bytes_received']:,}")
        
        print("\n❌ LOSSES:")
        print(f"   Wireless: {self.packets_lost_wireless:,} ({results['wireless_loss_rate']*100:.2f}%)")
        print(f"   Queue:    {self.packets_lost_queue:,} ({results['queue_loss_rate']*100:.2f}%)")
        print(f"   Timeouts: {results['timeouts']}")
        
        print("\n🔧 TCP STATE:")
        print(f"   Final CWND:     {results['final_cwnd']:.1f} packets")
        print(f"   Final SSThresh: {results['final_ssthresh']:.1f} packets")
        print(f"   Loss Events:    {results['loss_events']}")
        
        print(f"\n{'=' * 60}")
    
    def plot_results(self, filename: str = "simulation_results.png", show: bool = True):
        """Plot simulation results
        
        Args:
            filename: Output filename for the plot
            show: Whether to display the plot interactively
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('TCP Wireless Network Simulation Results', fontsize=14, fontweight='bold')
        
        results = self.get_results()
        net = self.config.network
        wireless = self.config.wireless
        
        # 1. Throughput over time
        ax = axes[0, 0]
        if len(self.throughput_history) > 2:
            times = [t for t, _ in self.throughput_history[1:]]
            throughputs = []
            for i in range(1, len(self.throughput_history)):
                t1, b1 = self.throughput_history[i-1]
                t2, b2 = self.throughput_history[i]
                if t2 > t1:
                    tp = (b2 - b1) * 8 / (t2 - t1) / 1e6
                    throughputs.append(max(0, tp))
                else:
                    throughputs.append(0)
            
            ax.plot(times, throughputs, linewidth=1.5, color='#2ecc71', alpha=0.8)
            if throughputs:
                ax.axhline(y=np.mean(throughputs), color='red', linestyle='--', 
                          label=f'Avg: {np.mean(throughputs):.2f} Mbps')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Throughput (Mbps)')
            ax.set_title('Throughput Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        
        # 2. Congestion Window
        ax = axes[0, 1]
        if self.cwnd_history:
            times = [t for t, _ in self.cwnd_history]
            cwnds = [c for _, c in self.cwnd_history]
            ax.plot(times, cwnds, linewidth=1.5, color='#3498db')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('CWND (packets)')
            ax.set_title('Congestion Window Evolution')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        
        # 3. RTT
        ax = axes[1, 0]
        if self.rtt_history:
            times = [t for t, _ in self.rtt_history]
            rtts = [r for _, r in self.rtt_history]
            ax.plot(times, rtts, linewidth=1, color='#e74c3c', alpha=0.6)
            ax.axhline(y=np.mean(rtts), color='blue', linestyle='--',
                      label=f'Avg: {np.mean(rtts):.2f} ms')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('RTT (milliseconds)')
            ax.set_title('RTT Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Summary panel
        ax = axes[1, 1]
        ax.axis('off')
        
        summary = f"""
        SIMULATION SUMMARY
        {'═' * 40}
        
        Network Configuration:
          • Bandwidth:    {net.bandwidth_mbps:.1f} Mbps
          • Base RTT:     {net.base_rtt_ms:.1f} ms
          • Loss Rate:    {wireless.loss_rate * 100:.1f}%
          • Queue Size:   {net.queue_size_packets} packets
        
        Results:
          • Throughput:   {results['throughput_mbps']:.2f} Mbps
          • Efficiency:   {results['efficiency_percent']:.1f}%
          • Avg RTT:      {results['avg_rtt_ms']:.2f} ms
          • Packets RX:   {results['packets_received']:,}
        
        Losses:
          • Wireless:     {self.packets_lost_wireless} packets
          • Queue Drops:  {self.packets_lost_queue} packets
          • Timeouts:     {results['timeouts']}
        """
        
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Ensure plots directory exists
        import os
        plots_dir = './plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # If filename doesn't include path, save to plots folder
        if os.path.dirname(filename) == '':
            filename = os.path.join(plots_dir, filename)
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()