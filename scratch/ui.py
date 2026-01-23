#!/usr/bin/env python3
"""
Gradio Web UI for TCP RL Congestion Control

Comprehensive interface for:
- Live simulation with parameter control
- Model training with custom parameters
- Analysis and plot generation
- Model comparison and evaluation

Author: RL-TCP Project
"""

import gradio as gr
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import threading

# Import project modules
from train_rl_agent import train_agent, evaluate_agent, compare_across_loss_rates
from tcp_rl_env import TCPWirelessEnv, RLConfig, evaluate_baseline_tcp, make_env
from python_tcp_simulator import NetworkConfig, WirelessConfig, TCPSimulator
from analyze_results import generate_all_plots, generate_eda_plots, generate_everything

# Global variables - use absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_available_models():
    """Get list of available trained models"""
    models = []
    if os.path.exists(MODELS_DIR):
        # Get all .zip files (saved models)
        zip_models = glob.glob(os.path.join(MODELS_DIR, "*.zip"))
        models.extend([os.path.basename(m).replace(".zip", "") for m in zip_models])
        
        # Also check for models without .zip extension (some SB3 models)
        # Look for files that match the pattern tcp_rl_*_final or tcp_rl_*_steps
        for file in os.listdir(MODELS_DIR):
            filepath = os.path.join(MODELS_DIR, file)
            if os.path.isfile(filepath) and not file.endswith('.zip'):
                # Check if it's a model file (has no extension or specific patterns)
                if 'tcp_rl_' in file and ('_final' in file or '_steps' in file):
                    if file not in models:  # Avoid duplicates
                        models.append(file)
    
    # Sort models by timestamp (most recent first)
    models.sort(reverse=True)
    return models if models else ["No models found"]


def get_available_plots():
    """Get list of available plots"""
    plots = []
    if os.path.exists(PLOTS_DIR):
        plots = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
        plots = [os.path.basename(p) for p in plots]
    return plots if plots else ["No plots found"]


def format_results(results_dict):
    """Format results dictionary as markdown"""
    md = "### 📊 Simulation Results\n\n"
    for key, value in results_dict.items():
        if isinstance(value, float):
            md += f"**{key.replace('_', ' ').title()}:** {value:.3f}\n\n"
        else:
            md += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
    return md


# ==============================================================================
# Tab 1: Live Simulation
# ==============================================================================

def run_live_simulation(bandwidth, rtt, loss_rate, queue_size, duration):
    """Run a live TCP simulation with given parameters"""
    try:
        # Log parameters to terminal
        print("\n" + "="*60)
        print("🎮 LIVE SIMULATION STARTED")
        print(f"   Bandwidth: {bandwidth} Mbps")
        print(f"   RTT: {rtt} ms")
        print(f"   Loss Rate: {loss_rate}% (decimal: {loss_rate/100.0})")
        print(f"   Queue Size: {queue_size} packets")
        print(f"   Duration: {duration} seconds")
        print("="*60)
        
        # Create simulator with custom parameters
        sim = TCPSimulator(
            bandwidth_mbps=bandwidth,
            rtt_ms=rtt,
            loss_rate=loss_rate / 100.0,  # Convert percentage to decimal
            queue_size=queue_size,
            simulation_time=duration
        )
        
        # Run simulation
        results = sim.run(verbose=False)
        
        # Log completion
        print(f"✅ Simulation complete! Throughput: {results['throughput_mbps']:.2f} Mbps, " 
              f"Actual Loss: {results['total_loss_rate']*100:.2f}%")
        print("="*60 + "\n")
        
        # Generate plot
        plot_path = os.path.join(PLOTS_DIR, "live_simulation.png")
        sim.plot_results(filename=plot_path, show=False)
        
        # Format results
        results_md = f"""
## 📊 Simulation Results

### Network Configuration
- **Bandwidth:** {bandwidth} Mbps
- **Base RTT:** {rtt} ms  
- **Wireless Loss Rate:** {loss_rate}%
- **Queue Size:** {queue_size} packets
- **Duration:** {duration} seconds

### Performance Metrics
- **Average Throughput:** {results['throughput_mbps']:.2f} Mbps
- **Link Efficiency:** {results['efficiency_percent']:.1f}%
- **Average RTT:** {results['avg_rtt_ms']:.2f} ms
- **Packets Received:** {results['packets_received']:,}

### Loss Statistics
- **Wireless Losses:** {results['packets_lost_wireless']} packets
- **Queue Drops:** {results['packets_lost_queue']} packets
- **Timeouts:** {results['timeouts']}
- **Total Loss Rate:** {results['total_loss_rate']*100:.2f}%

### TCP State
- **Final CWND:** {results['final_cwnd']:.1f} packets
- **Final SSThresh:** {results['final_ssthresh']:.1f} packets
"""
        
        return plot_path, results_md
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ==============================================================================
# Tab 2: Train New Model
# ==============================================================================

def train_new_model(timesteps, loss_rate, bandwidth, rtt, algorithm, seed, progress=gr.Progress()):
    """Train a new RL model with custom parameters"""
    try:
        # Log parameters to terminal
        print("\n" + "="*60)
        print("🤖 TRAINING NEW MODEL")
        print(f"   Algorithm: {algorithm}")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Loss Rate: {loss_rate}% (decimal: {loss_rate/100.0})")
        print(f"   Bandwidth: {bandwidth} Mbps")
        print(f"   RTT: {rtt} ms")
        print(f"   Seed: {seed}")
        print("="*60)
        
        progress(0, desc="Initializing training...")
        
        # Train model
        model, info = train_agent(
            total_timesteps=timesteps,
            loss_rate=loss_rate / 100.0,
            bandwidth_mbps=bandwidth,
            rtt_ms=rtt,
            algorithm=algorithm,
            save_path=MODELS_DIR,
            log_path="./logs",
            seed=seed,
            verbose=1
        )
        
        progress(1.0, desc="Training complete!")
        
        # Log actual results to terminal
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED")
        print(f"   Baseline Throughput: {info['baseline_throughput']:.2f} Mbps")
        print(f"   RL Agent Throughput: {info['final_throughput']:.2f} Mbps")
        print(f"   Improvement: {info['improvement_percent']:+.1f}%")
        print(f"   Training Time: {info['training_time']:.1f}s")
        print("="*60 + "\n")
        
        # Format training summary
        summary = f"""
## 🎉 Training Complete!

### Training Configuration
- **Algorithm:** {algorithm}
- **Total Timesteps:** {timesteps:,}
- **Loss Rate:** {loss_rate}%
- **Bandwidth:** {bandwidth} Mbps
- **RTT:** {rtt} ms
- **Seed:** {seed}

### Results
- **Training Time:** {info['training_time']:.1f} seconds
- **Mean Reward:** {info['mean_reward']:.3f} ± {info['std_reward']:.3f}
- **Baseline Throughput:** {info['baseline_throughput']:.2f} Mbps
- **RL Agent Throughput:** {info['final_throughput']:.2f} Mbps
- **Improvement:** {info['improvement_percent']:+.1f}%

### Model Saved
📁 **Path:** `{info['model_path']}`

You can now use this model in the "Test Model" tab!
"""
        
        # Get training plot if it exists
        training_plots = glob.glob(os.path.join(PLOTS_DIR, "*_training.png"))
        training_plot = training_plots[-1] if training_plots else None
        
        return summary, training_plot, gr.update(choices=get_available_models())
        
    except Exception as e:
        return f"❌ Error during training: {str(e)}", None, gr.update()


# ==============================================================================
# Tab 3: Test Model
# ==============================================================================

def test_model(model_name, loss_rate, bandwidth, rtt, n_episodes):
    """Evaluate a trained model"""
    try:
        if model_name == "No models found" or not model_name:
            return "⚠️ Please train a model first or select an existing model.", None
        
        # Log parameters to terminal
        print("\n" + "="*60)
        print("📊 TESTING MODEL")
        print(f"   Model: {model_name}")
        print(f"   Loss Rate: {loss_rate}% (decimal: {loss_rate/100.0})")
        print(f"   Bandwidth: {bandwidth} Mbps")
        print(f"   RTT: {rtt} ms")
        print(f"   Episodes: {n_episodes}")
        print("="*60)
        
        model_path = os.path.join(MODELS_DIR, model_name)
        print(f"Full model path: {model_path}")
        
        # Evaluate model
        results = evaluate_agent(
            model_path=model_path,
            loss_rate=loss_rate / 100.0,
            bandwidth_mbps=bandwidth,
            rtt_ms=rtt,
            n_episodes=n_episodes,
            render=False
        )
        
        # Log results to terminal
        print("\n" + "="*60)
        print("✅ TEST RESULTS")
        print(f"   Baseline Throughput: {results['baseline']['avg_throughput']:.2f} Mbps")
        print(f"   RL Agent Throughput: {results['rl_throughput']:.2f} Mbps")
        print(f"   Improvement: {results['improvement']:+.1f}%")
        print("="*60 + "\n")
        
        # Format results
        results_md = f"""
## 🤖 Model Evaluation Results

### Test Configuration
- **Model:** {model_name}
- **Episodes:** {n_episodes}
- **Loss Rate:** {loss_rate}%
- **Bandwidth:** {bandwidth} Mbps
- **RTT:** {rtt} ms

### Performance Comparison

| Metric | TCP Baseline | RL Agent | Improvement |
|--------|--------------|----------|-------------|
| **Throughput** | {results['baseline']['avg_throughput']:.2f} Mbps | {results['rl_throughput']:.2f} Mbps | **{results['improvement']:+.1f}%** |
| **RTT** | {results['baseline']['avg_rtt']:.2f} ms | {results['rl_rtt']:.2f} ms | - |
| **Reward** | {results['baseline']['avg_reward']:.3f} | {results['rl_reward']:.3f} | - |

### 🎯 Key Insights
"""
        
        if results['improvement'] > 20:
            results_md += "\n✅ **Excellent!** RL agent shows significant improvement over TCP.\n"
        elif results['improvement'] > 0:
            results_md += "\n✅ **Good!** RL agent outperforms TCP baseline.\n"
        else:
            results_md += "\n⚠️ **Note:** RL agent performance is similar to or below TCP baseline.\n"
        
        return results_md, None
        
    except Exception as e:
        return f"❌ Error during evaluation: {str(e)}", None


def compare_models_across_loss(model_name, progress=gr.Progress()):
    """Compare model performance across different loss rates"""
    try:
        if model_name == "No models found" or not model_name:
            return "⚠️ Please select a model first.", None
        
        # Log parameters to terminal
        print("\n" + "="*60)
        print("📈 COMPARING ACROSS LOSS RATES")
        print(f"   Model: {model_name}")
        print(f"   Loss Rates: 1%, 2%, 3%, 5%, 10%")
        print("="*60)
        
        progress(0, desc="Running comparison...")
        model_path = os.path.join(MODELS_DIR, model_name)
        
        # Run comparison
        results = compare_across_loss_rates(
            model_path=model_path,
            loss_rates=[0.01, 0.02, 0.03, 0.05, 0.10],
            n_episodes=3
        )
        
        progress(1.0, desc="Comparison complete!")
        
        # Format results
        results_md = "## 📈 Performance Across Loss Rates\n\n"
        results_md += "| Loss Rate | TCP (Mbps) | RL Agent (Mbps) | Improvement |\n"
        results_md += "|-----------|------------|-----------------|-------------|\n"
        
        for r in results:
            results_md += f"| {r['loss_rate']*100:.0f}% | {r['tcp_throughput']:.2f} | {r['rl_throughput']:.2f} | **{r['improvement']:+.1f}%** |\n"
        
        # Get comparison plot
        comp_plot = os.path.join(PLOTS_DIR, "rl_vs_tcp_comparison.png")
        
        return results_md, comp_plot if os.path.exists(comp_plot) else None
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None


# ==============================================================================
# Tab 4: Generate Plots
# ==============================================================================

def generate_analysis_plots(model_name, plot_types, progress=gr.Progress()):
    """Generate analysis plots"""
    try:
        if model_name == "No models found" or not model_name:
            return "⚠️ Please select a model first.", []
        
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if "All Analysis Plots" in plot_types:
            progress(0, desc="Generating all analysis plots...")
            generate_all_plots(model_path, PLOTS_DIR)
            
        if "EDA Plots" in plot_types:
            progress(0.5, desc="Generating EDA plots...")
            generate_eda_plots(PLOTS_DIR)
        
        if "Everything" in plot_types:
            progress(0, desc="Generating ALL plots...")
            generate_everything(model_path, PLOTS_DIR)
        
        progress(1.0, desc="Plots generated!")
        
        # Get generated plots
        plot_files = get_available_plots()
        plot_paths = [os.path.join(PLOTS_DIR, p) for p in plot_files]
        
        success_msg = f"✅ Successfully generated {len(plot_files)} plots!\n\nPlots saved to: `{PLOTS_DIR}`"
        
        return success_msg, plot_paths[:5]  # Show first 5 plots
        
    except Exception as e:
        return f"❌ Error: {str(e)}", []


def view_plot(plot_name):
    """View a specific plot"""
    if plot_name and plot_name != "No plots found":
        plot_path = os.path.join(PLOTS_DIR, plot_name)
        if os.path.exists(plot_path):
            return plot_path
    return None


# ==============================================================================
# Tab 5: Configuration & Info
# ==============================================================================

def get_config_info():
    """Get current configuration information"""
    config_md = """
# ⚙️ Project Configuration

## Current Setup
- **Models Directory:** `./models`
- **Plots Directory:** `./plots`
- **Logs Directory:** `./logs`

## Default Parameters

### Network Configuration
- **Bandwidth:** 10.0 Mbps
- **Base RTT:** 40.0 ms
- **Queue Size:** 100 packets
- **MTU:** 1500 bytes

### Wireless Configuration
- **Loss Rate:** 2% (0.02)

### RL Configuration
- **Throughput Weight:** 2.0
- **Latency Weight:** 0.0
- **Loss Weight:** 0.0 (key for wireless!)
- **Min CWND:** 10.0
- **Max CWND:** 150.0

## Training Defaults
- **Algorithm:** PPO
- **Timesteps:** 30,000
- **Learning Rate:** 3e-4
- **Batch Size:** 64

## Key Insights
- ✅ **Loss Weight = 0.0** allows agent to learn wireless loss ≠ congestion
- ✅ **Throughput Weight = 2.0** provides stable training
- ✅ **Latency Weight = 0.0** maximizes throughput focus

---

For more details, see the documentation files:
- `EDA_GUIDE.md`
- `EDA_QUICK_REFERENCE.md`
- `EDA_RESULTS_README.md`
"""
    return config_md


# ==============================================================================
# Build Gradio Interface
# ==============================================================================

def build_ui():
    """Build the complete Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .tab-nav button {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="TCP RL Control") as demo:
        
        # Header
        gr.Markdown("""
        # 🌐 TCP RL Congestion Control for Wireless Networks
        
        ### Reinforcement Learning Solution for 5G/Wi-Fi TCP Optimization
        
        **Problem:** Standard TCP interprets all packet loss as congestion, causing unnecessary throughput reduction in wireless networks.
        
        **Solution:** RL agent learns to distinguish between wireless loss and actual congestion, maintaining higher throughput.
        
        ---
        """)
        
        # Tabs
        with gr.Tabs():
            
            # ============ Tab 1: Live Simulation ============
            with gr.Tab("🎮 Live Simulation"):
                gr.Markdown("### Run TCP Simulation with Custom Parameters")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Network Parameters")
                        sim_bandwidth = gr.Slider(1, 100, value=10, step=1, label="Bandwidth (Mbps)")
                        sim_rtt = gr.Slider(10, 500, value=40, step=10, label="Base RTT (ms)")
                        sim_loss = gr.Slider(0, 20, value=2, step=0.5, label="Wireless Loss Rate (%)")
                        sim_queue = gr.Slider(10, 500, value=100, step=10, label="Queue Size (packets)")
                        sim_duration = gr.Slider(5, 120, value=30, step=5, label="Simulation Duration (s)")
                        
                        sim_run_btn = gr.Button("🚀 Run Simulation", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Results")
                        sim_results = gr.Markdown()
                        sim_plot = gr.Image(label="Simulation Plots")
                
                sim_run_btn.click(
                    run_live_simulation,
                    inputs=[sim_bandwidth, sim_rtt, sim_loss, sim_queue, sim_duration],
                    outputs=[sim_plot, sim_results]
                )
            
            # ============ Tab 2: Train Model ============
            with gr.Tab("🤖 Train New Model"):
                gr.Markdown("### Train RL Agent with Custom Parameters")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Training Configuration")
                        train_timesteps = gr.Slider(5000, 100000, value=30000, step=1000, 
                                                    label="Training Timesteps")
                        train_algorithm = gr.Dropdown(["PPO", "SAC", "TD3"], value="PPO", 
                                                     label="RL Algorithm")
                        train_seed = gr.Number(value=42, label="Random Seed", precision=0)
                        
                        gr.Markdown("#### Network Environment")
                        train_loss = gr.Slider(0, 20, value=2, step=0.5, label="Wireless Loss Rate (%)")
                        train_bandwidth = gr.Slider(1, 50, value=10, step=1, label="Bandwidth (Mbps)")
                        train_rtt = gr.Slider(10, 200, value=40, step=10, label="Base RTT (ms)")
                        
                        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Training Progress & Results")
                        train_results = gr.Markdown()
                        train_plot = gr.Image(label="Training Progress")
                
                # Hidden output to update model dropdown
                model_dropdown_update = gr.State()
                
                train_btn.click(
                    train_new_model,
                    inputs=[train_timesteps, train_loss, train_bandwidth, train_rtt, 
                           train_algorithm, train_seed],
                    outputs=[train_results, train_plot, model_dropdown_update]
                )
            
            # ============ Tab 3: Test Model ============
            with gr.Tab("📊 Test Model"):
                gr.Markdown("### Evaluate Trained Models")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Model Selection")
                        test_model_dropdown = gr.Dropdown(choices=get_available_models(), 
                                                         label="Select Model",
                                                         value=get_available_models()[0] if get_available_models() else None)
                        refresh_models_btn = gr.Button("🔄 Refresh Model List")
                        
                        gr.Markdown("#### Test Configuration")
                        test_loss = gr.Slider(0, 20, value=2, step=0.5, label="Wireless Loss Rate (%)")
                        test_bandwidth = gr.Slider(1, 50, value=10, step=1, label="Bandwidth (Mbps)")
                        test_rtt = gr.Slider(10, 200, value=40, step=10, label="Base RTT (ms)")
                        test_episodes = gr.Slider(1, 20, value=5, step=1, label="Number of Episodes")
                        
                        test_btn = gr.Button("🧪 Run Evaluation", variant="primary")
                        compare_btn = gr.Button("📈 Compare Across Loss Rates", variant="secondary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Evaluation Results")
                        test_results = gr.Markdown()
                        test_plot = gr.Image(label="Comparison Plot")
                
                refresh_models_btn.click(
                    lambda: gr.update(choices=get_available_models()),
                    outputs=test_model_dropdown
                )
                
                test_btn.click(
                    test_model,
                    inputs=[test_model_dropdown, test_loss, test_bandwidth, test_rtt, test_episodes],
                    outputs=[test_results, test_plot]
                )
                
                compare_btn.click(
                    compare_models_across_loss,
                    inputs=[test_model_dropdown],
                    outputs=[test_results, test_plot]
                )
            
            # ============ Tab 4: Generate Plots ============
            with gr.Tab("📈 Generate Plots"):
                gr.Markdown("### Generate Analysis & EDA Plots")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Plot Generation")
                        plot_model_dropdown = gr.Dropdown(choices=get_available_models(), 
                                                         label="Select Model",
                                                         value=get_available_models()[0] if get_available_models() else None)
                        
                        plot_types = gr.CheckboxGroup(
                            ["All Analysis Plots", "EDA Plots", "Everything"],
                            value=["All Analysis Plots"],
                            label="Plot Types to Generate"
                        )
                        
                        generate_btn = gr.Button("🎨 Generate Plots", variant="primary", size="lg")
                        
                        gr.Markdown("#### View Existing Plots")
                        view_plot_dropdown = gr.Dropdown(choices=get_available_plots(), 
                                                        label="Select Plot to View")
                        refresh_plots_btn = gr.Button("🔄 Refresh Plot List")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Generated Plots")
                        plot_status = gr.Markdown()
                        plot_gallery = gr.Gallery(label="Preview", columns=2, height=400)
                        single_plot_view = gr.Image(label="Selected Plot")
                
                generate_btn.click(
                    generate_analysis_plots,
                    inputs=[plot_model_dropdown, plot_types],
                    outputs=[plot_status, plot_gallery]
                )
                
                refresh_plots_btn.click(
                    lambda: gr.update(choices=get_available_plots()),
                    outputs=view_plot_dropdown
                )
                
                view_plot_dropdown.change(
                    view_plot,
                    inputs=view_plot_dropdown,
                    outputs=single_plot_view
                )
            
            # ============ Tab 5: Configuration & Info ============
            with gr.Tab("⚙️ Configuration & Info"):
                gr.Markdown(get_config_info())
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📁 Project Files")
                        gr.File(label="Upload Model", file_types=[".zip"])
                        
                    with gr.Column():
                        gr.Markdown("### 📊 Quick Stats")
                        num_models = len(get_available_models())
                        num_plots = len(get_available_plots())
                        gr.Markdown(f"""
                        - **Available Models:** {num_models}
                        - **Generated Plots:** {num_plots}
                        - **Plots Directory:** `{PLOTS_DIR}`
                        - **Models Directory:** `{MODELS_DIR}`
                        """)
        
        # Footer
        gr.Markdown("""
        ---
        ### 📚 Documentation
        - See `EDA_GUIDE.md` for parameter optimization insights
        - See `EDA_RESULTS_README.md` for detailed configuration analysis
        - See `EDA_QUICK_REFERENCE.md` for quick parameter reference
        
        ### 🚀 Key Features
        - ✅ Live TCP simulation with custom parameters
        - ✅ Train RL agents with PPO/SAC/TD3 algorithms
        - ✅ Comprehensive model evaluation
        - ✅ Automatic plot generation
        - ✅ Parameter sensitivity analysis (EDA)
        
        **Made with ❤️ for TCP optimization in wireless networks**
        """)
    
    return demo


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" TCP RL Congestion Control - Web Interface")
    print("=" * 60)
    print("\nStarting Gradio interface...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Plots directory: {PLOTS_DIR}")
    print(f"Available models: {len(get_available_models())}")
    print(f"Available plots: {len(get_available_plots())}")
    print("\n" + "=" * 60)
    
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",  # Local access only
        server_port=None,  # Auto-select available port
        share=False,  # Set to True to get public link
        show_error=True
    )
