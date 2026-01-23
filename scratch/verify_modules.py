#!/usr/bin/env python3
"""
Verification script for modular architecture
"""

import sys

print('='*60)
print('MODULAR ARCHITECTURE VERIFICATION')
print('='*60)

# Test 1: Direct module imports
print('\n[1] Testing direct module imports...')
try:
    from config import SimulatorConfig, NetworkConfig, TCPConfig, WirelessConfig
    from packet import Packet
    from simulator import TCPSimulator
    from analysis import compare_loss_rates, print_comparison_summary
    from cli import parse_args, main
    print('    ✓ All module imports successful')
except Exception as e:
    print(f'    ✗ Import failed: {e}')
    sys.exit(1)

# Test 2: Backward compatibility imports
print('\n[2] Testing backward compatibility imports...')
try:
    from python_tcp_simulator import TCPSimulator as TCPSim2
    from python_tcp_simulator import NetworkConfig as NetConfig2
    print('    ✓ Backward compatibility wrapper working')
except Exception as e:
    print(f'    ✗ Import failed: {e}')
    sys.exit(1)

# Test 3: Package import
print('\n[3] Testing package imports...')
try:
    from __init__ import TCPSimulator as TCPSim3
    print('    ✓ Package __init__.py working')
except Exception as e:
    print(f'    ✗ Import failed: {e}')
    sys.exit(1)

# Test 4: Create simulator instance
print('\n[4] Testing simulator instantiation...')
try:
    sim = TCPSimulator(bandwidth_mbps=10.0, loss_rate=0.02, simulation_time=5.0)
    print('    ✓ TCPSimulator instance created')
except Exception as e:
    print(f'    ✗ Instantiation failed: {e}')
    sys.exit(1)

# Test 5: Verify configuration
print('\n[5] Verifying configuration...')
try:
    assert sim.config.network.bandwidth_mbps == 10.0
    assert sim.config.wireless.loss_rate == 0.02
    assert sim.config.simulation.duration_sec == 5.0
    print('    ✓ Configuration verified')
except Exception as e:
    print(f'    ✗ Verification failed: {e}')
    sys.exit(1)

# Test 6: Quick simulation run
print('\n[6] Running quick simulation (5 seconds)...')
try:
    results = sim.run(verbose=False)
    print('    ✓ Simulation completed')
    print('    - Throughput: {:.2f} Mbps'.format(results['throughput_mbps']))
    print('    - Packets RX: {}'.format(results['packets_received']))
    print('    - Efficiency: {:.1f}%'.format(results['efficiency_percent']))
except Exception as e:
    print('    ✗ Simulation failed: {}'.format(e))
    sys.exit(1)

# Test 7: Verify RL environment integration
print('\n[7] Testing RL environment integration...')
try:
    from tcp_rl_env import TCPWirelessEnv
    env = TCPWirelessEnv()
    print('    ✓ TCPWirelessEnv initialized with new modules')
except Exception as e:
    print('    ✗ Integration failed: {}'.format(e))
    sys.exit(1)

print('\n' + '='*60)
print('✅ ALL TESTS PASSED - MODULAR ARCHITECTURE WORKING')
print('='*60)
