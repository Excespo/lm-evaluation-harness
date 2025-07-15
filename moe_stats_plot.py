import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

def create_task_heatmap(json_data, mapping_file, task_name, output_dir=None):
    """
    Create heat map for a single task showing all layers and experts
    
    Parameters:
    json_data: dict - JSON data for the task (layer_id: stats_list)
    mapping_file: str or dict - mapping file path or mapping dictionary  
    task_name: str - name of the task
    output_dir: str - output directory for saving plots
    """
    
    # Read mapping file
    if isinstance(mapping_file, str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    else:
        mapping = mapping_file
    
    # Convert JSON data to a 2D matrix: layers x experts
    layer_names = list(json_data.keys())
    layer_names.sort()  # Sort layer names for consistent ordering
    
    if not layer_names:
        print(f"No data found for task {task_name}")
        return
        
    # Get the number of experts from the first layer
    first_layer_data = json_data[layer_names[0]]
    n_experts = len(first_layer_data)
    n_layers = len(layer_names)
    
    # Create matrix: rows=layers, cols=experts
    stats_matrix = np.zeros((n_layers, n_experts))
    
    for i, layer_name in enumerate(layer_names):
        layer_data = json_data[layer_name]
        if len(layer_data) != n_experts:
            print(f"Warning: Layer {layer_name} has {len(layer_data)} experts, expected {n_experts}")
            # Pad or truncate to match expected size
            if len(layer_data) < n_experts:
                layer_data.extend([0] * (n_experts - len(layer_data)))
            else:
                layer_data = layer_data[:n_experts]
        stats_matrix[i, :] = layer_data
    
    # Convert to percentages (each row sums to 100%)
    percentage_matrix = np.zeros_like(stats_matrix)
    for i in range(n_layers):
        row_sum = np.sum(stats_matrix[i, :])
        if row_sum > 0:
            percentage_matrix[i, :] = (stats_matrix[i, :] / row_sum) * 100
        else:
            percentage_matrix[i, :] = 0
    
    # Create the heatmap
    plt.figure(figsize=(max(12, n_experts), max(8, n_layers)))
    
    # Create labels for axes
    expert_labels = [f"Expert_{i}" for i in range(n_experts)]
    if 'labels' in mapping and len(mapping['labels']) == n_experts:
        expert_labels = mapping['labels']
    
    # Create heatmap
    sns.heatmap(percentage_matrix, 
                xticklabels=expert_labels,
                yticklabels=layer_names,
                annot=True, 
                cmap='viridis', 
                fmt='.1f',
                cbar_kws={'label': 'Usage Percentage (%)'})
    
    plt.title(f'MoE Expert Usage Statistics - Task: {task_name}')
    plt.xlabel('Experts')
    plt.ylabel('Layers')
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = os.getcwd()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"moe_stats_{task_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved MoE statistics for task '{task_name}' to: {filepath}")

def process_moe_stats_file(json_file, mapping_file):
    """
    Process MoE stats file and create visualizations
    
    Parameters:
    json_file: str - path to the MoE stats JSON file
    mapping_file: str - path to the mapping file
    """
    
    # Get output directory (same as JSON file directory)
    output_dir = os.path.dirname(json_file)
    
    # Extract task name from file path
    # Example: .../nq_open_history/.../moe_stats_xxx.jsonl -> nq_open_history
    path_parts = json_file.split(os.path.sep)
    task_name = "unknown_task"
    
    for part in path_parts:
        if "nq_open" in part or "trivia" in part or "natural_questions" in part or "_history" in part:
            task_name = part
            break
    
    # If still unknown, try to extract from filename
    if task_name == "unknown_task":
        filename = os.path.basename(json_file)
        if "moe_stats" in filename:
            task_name = filename.replace("moe_stats_", "").replace(".jsonl", "").replace(".json", "")
    
    try:
        # Read JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            if json_file.endswith('.jsonl'):
                # Handle JSONL format (one JSON object per line)
                lines = f.readlines()
                if len(lines) == 1:
                    raw_data = json.loads(lines[0])
                else:
                    # If multiple lines, combine them or take the last one
                    raw_data = json.loads(lines[-1])  # Take the last line
            else:
                # Handle regular JSON format
                raw_data = json.load(f)
        
        # Extract the actual stats data and task name from the JSON structure
        if isinstance(raw_data, dict):
            if 'stats' in raw_data:
                # Format: {"task": "...", "stats": {"0": [...], "1": [...]}, "timestamp": "..."}
                json_data = raw_data['stats']
                if 'task' in raw_data:
                    task_name = raw_data['task']
            elif 'task' in raw_data:
                # Handle other possible formats
                json_data = {k: v for k, v in raw_data.items() if k not in ['task', 'timestamp']}
            else:
                # Assume the entire dict is layer data
                json_data = raw_data
        else:
            json_data = raw_data
        
        print(f"Processing task: {task_name}")
        print(f"Found {len(json_data)} layers in the data")
        
        # Create heatmap for this task
        create_task_heatmap(json_data, mapping_file, task_name, output_dir)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Error processing file: {e}")
        raise

# Example usage function
def example_usage():
    """
    Example code showing how to use
    """
    # Example JSON data (simulating task data with multiple layers)
    sample_json_data = {
        "layer_0": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9],
        "layer_5": [9.1, 8.2, 7.3, 6.4, 5.5, 4.6, 3.7, 2.8],
        "layer_10": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        "layer_15": [2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8]
    }
    
    # Example mapping data
    sample_mapping = {
        "labels": ["Expert_A", "Expert_B", "Expert_C", "Expert_D", "Expert_E", "Expert_F", "Expert_G", "Expert_H"],
        "description": "Sample mapping for demonstration"
    }
    
    print("Creating task heatmap for example data:")
    create_task_heatmap(sample_json_data, sample_mapping, "example_task")

# Main function
def main():
    """
    Main function - process the MoE stats file
    """
    # Your JSON file path
    json_file = "/aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/outputs/verify_stats/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS_final/hf/use_history/nq_open_history/__aistor__sjtu__hpc_stor01__home__luoyijie__ckpts__fpm__Qwen2.5-MoE-5.7B-wiki-mean-common-32BS__final/moe_stats_2025-06-26T15-27-44.002187.jsonl"
    mapping_file = "/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS/final/function_to_expert.index.json"
    
    try:
        print("Processing MoE statistics file...")
        process_moe_stats_file(json_file, mapping_file)
        print("Done!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Running example code:")
        example_usage()
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Running example code:")
        example_usage()

if __name__ == "__main__":
    # Run main function to process MoE stats
    main()