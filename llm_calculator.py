import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_vram_requirements(parameters_billions, precision):
    """Calculate VRAM requirements for different model sizes and precisions"""
    if precision == "FP16":
        return parameters_billions * 2  # 2 bytes per parameter
    elif precision == "INT8":
        return parameters_billions * 1  # 1 byte per parameter
    elif precision == "INT4":
        return parameters_billions * 0.5  # 0.5 bytes per parameter
    else:
        return parameters_billions * 4  # FP32 (4 bytes per parameter)

def estimate_throughput(parameters_billions, vram_gb, gpu_type):
    """Estimate throughput in tokens per second based on model size and GPU"""
    
    # Base throughput values derived from benchmarks
    if gpu_type == "RTX 3090":
        base_throughput = 100
        vram_factor = 1.0
    elif gpu_type == "RTX 4090":
        base_throughput = 140
        vram_factor = 1.3
    elif gpu_type == "A5000":
        base_throughput = 110
        vram_factor = 1.2
    elif gpu_type == "A100":
        base_throughput = 180
        vram_factor = 1.5
    else:  # Default/fallback
        base_throughput = 80
        vram_factor = 0.8
    
    # Model size impact (larger models are slower)
    size_factor = 1 / (1 + 0.1 * parameters_billions)
    
    # VRAM utilization impact (more VRAM headroom = better performance)
    vram_utilization = min(1.0, parameters_billions * 2 / vram_gb)
    vram_headroom = max(0.5, 1 - vram_utilization)
    
    throughput = base_throughput * size_factor * vram_factor * vram_headroom
    return max(5, int(throughput))  # Ensure minimum throughput of 5 tokens/sec

def main():
    st.title("Local LLM Deployment Calculator")
    st.write("Estimate resource requirements and performance for local LLMs")
    
    # Sidebar for configuration inputs
    st.sidebar.header("Hardware Configuration")
    gpu_type = st.sidebar.selectbox(
        "GPU Type",
        ["RTX 3090", "RTX 4090", "A5000", "A100", "NVIDIA DIGITS", "Ascend 910B"]
    )
    
    if gpu_type == "RTX 3090":
        vram_gb = 24
    elif gpu_type == "RTX 4090":
        vram_gb = 24
    elif gpu_type == "A5000":
        vram_gb = 24
    elif gpu_type == "A100":
        vram_gb = 40
    elif gpu_type == "NVIDIA DIGITS":
        vram_gb = 128
    elif gpu_type == "Ascend 910B":
        vram_gb = 32
    
    vram_gb = st.sidebar.slider("VRAM (GB)", min_value=8, max_value=80, value=vram_gb)
    
    ram_gb = st.sidebar.slider("System RAM (GB)", min_value=16, max_value=512, value=128, step=16)
    
    st.sidebar.header("Model Configuration")
    precision = st.sidebar.selectbox(
        "Quantization Precision",
        ["FP32", "FP16", "INT8", "INT4"]
    )
    
    num_gpus = st.sidebar.slider("Number of GPUs", min_value=1, max_value=4, value=1)
    
    total_vram = vram_gb * num_gpus
    
    # Main area calculations
    st.header("Model Size Capabilities")
    
    model_sizes = pd.DataFrame({
        "Parameters (billions)": [1, 2, 3, 7, 13, 34, 70],
        "Model Examples": [
            "Tiny models (Phi-1.5)",
            "Small models (CT-LLM, Phi-2)",
            "Efficient models (Qwen 2.5-3B)",
            "Base models (Mistral 7B, Code Llama 7B)",
            "Mid-size models (Llama 2 13B, Qwen 2.5-14B)",
            "Large models (Code Llama 34B)",
            "Very large models (Llama 2 70B, Qwen 2.5-72B)"
        ]
    })
    
    # Calculate VRAM requirements for each model size
    model_sizes["VRAM Required (GB)"] = model_sizes["Parameters (billions)"].apply(
        lambda x: calculate_vram_requirements(x, precision)
    )
    
    model_sizes["Fits in Available VRAM"] = model_sizes["VRAM Required (GB)"] <= total_vram
    model_sizes["Estimated Throughput (tokens/sec)"] = model_sizes.apply(
        lambda row: estimate_throughput(row["Parameters (billions)"], total_vram, gpu_type) 
        if row["Fits in Available VRAM"] else 0, 
        axis=1
    )
    
    # Display the model size capabilities
    st.dataframe(model_sizes)
    
    # Visualize model sizes and VRAM requirements
    st.header("VRAM Requirements by Model Size")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        model_sizes["Model Examples"], 
        model_sizes["VRAM Required (GB)"],
        color=[
            'green' if fits else 'red' 
            for fits in model_sizes["Fits in Available VRAM"]
        ]
    )
    
    # Add a horizontal line for available VRAM
    ax.axhline(y=total_vram, color='blue', linestyle='-', label=f'Available VRAM ({total_vram} GB)')
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('VRAM Required (GB)')
    plt.title(f'VRAM Requirements with {precision} Precision')
    plt.tight_layout()
    plt.legend()
    
    st.pyplot(fig)
    
    # Throughput comparison
    st.header("Performance Estimates")
    
    viable_models = model_sizes[model_sizes["Fits in Available VRAM"]]
    
    if not viable_models.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars2 = ax2.bar(
            viable_models["Model Examples"], 
            viable_models["Estimated Throughput (tokens/sec)"],
            color='skyblue'
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Tokens per Second')
        plt.title(f'Estimated Throughput for {gpu_type} with {num_gpus} GPU(s)')
        plt.tight_layout()
        
        st.pyplot(fig2)
        
        # Practical implications
        st.header("Practical Capabilities")
        
        max_model_size = viable_models["Parameters (billions)"].max()
        max_throughput = viable_models["Estimated Throughput (tokens/sec)"].max()
        
        st.write(f"**Maximum Viable Model Size:** {max_model_size} billion parameters")
        st.write(f"**Peak Throughput:** {max_throughput} tokens per second")
        
        # Estimated capabilities
        st.write("### Estimated Use Case Support")
        
        if max_model_size >= 34:
            st.write("✅ **Enterprise-grade translation** (comparable to commercial services)")
        elif max_model_size >= 13:
            st.write("✅ **High-quality translation** (suitable for most business documents)")
        elif max_model_size >= 7:
            st.write("✅ **Good translation** (adequate for general business use)")
        else:
            st.write("⚠️ **Basic translation** (may struggle with complex content)")
        
        if max_model_size >= 34:
            st.write("✅ **Advanced code generation** (complex functions and algorithms)")
        elif max_model_size >= 13:
            st.write("✅ **Strong code generation** (complete function implementation)")
        elif max_model_size >= 7:
            st.write("✅ **Good code assistance** (syntax completion, simple functions)")
        else:
            st.write("⚠️ **Basic code completion** (limited to simple snippets)")
        
        # Concurrent users estimate
        concurrent_users = max(1, int(max_throughput / 20))  # Rough estimate: 20 tokens/sec per user
        st.write(f"**Estimated Concurrent Users:** {concurrent_users}")
    else:
        st.error("No viable models found for the current configuration. Try increasing VRAM or using a higher level of quantization.")

if __name__ == "__main__":
    main()