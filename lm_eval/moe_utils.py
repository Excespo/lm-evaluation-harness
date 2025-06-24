import re
import json
from typing import Optional, Any
import torch


def test_hook(module, input, output):
    print(f"Module {module.__class__.__name__} triggered")
    print(f"Input shape {input[0].shape}")
    if isinstance(output, tuple) and len(output) > 1 and hasattr(output[1], 'shape'):
        print(f"Output combine_weights shape {output[1].shape}")
    else:
        print(f"Output: {output}")
    return
    


def register_moe_hooks(model):
    """Register forward hooks on all TopKGate modules in the model."""
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, "__class__") and "TopKGate" in module.__class__.__name__:
            hook = module.register_forward_hook(test_hook)
            hooks.append(hook)
            # print(f"Registered MoE statistics hook on: {name}")
    return hooks


def unregister_moe_hooks(hooks):
    """Remove all registered hooks."""
    for hook in hooks:
        hook.remove()


def create_moe_stats():
    """Creates a new, empty dictionary to store MoE expert statistics."""
    return {}


def add_moe_stats_entry(
    stats_collector, doc_id, layer_id, token_id, dispatched_expert_ids
):
    """Add an expert activation entry to the statistics collector."""
    if doc_id not in stats_collector:
        stats_collector[doc_id] = {}
    if layer_id not in stats_collector[doc_id]:
        stats_collector[doc_id][layer_id] = {}
    stats_collector[doc_id][layer_id][token_id] = dispatched_expert_ids


def load_moe_stats(path):
    """Load statistics from a JSON file."""
    with open(path, "rt", encoding="utf-8") as fin:
        return json.load(fin)


def save_moe_stats(stats_collector, path, indent=4):
    """Save statistics to a JSON file."""
    with open(path, "wt", encoding="utf-8") as fout:
        json.dump(stats_collector, fout, ensure_ascii=False, indent=indent)


# Global variables for hook-based statistics collection
_moe_stats_data = None
_current_doc_id = None
_current_token_offset = 0


def _record_expert_stats(
    statistics_collector: Optional[dict],
    doc_id: Optional[Any],
    layer_id: int,
    combine_weights: torch.Tensor,
    token_offset: int = 0,
):
    """Record expert activation statistics from combine_weights tensor."""
    if statistics_collector is None or doc_id is None:
        return

    if combine_weights.dim() == 3:
        active_expert_weights = torch.sum(combine_weights, dim=2)
    else:
        active_expert_weights = combine_weights
    
    active_mask = active_expert_weights > 1e-6
    token_indices_gpu, expert_indices_gpu = torch.where(active_mask)

    token_indices_cpu = token_indices_gpu.cpu().tolist()
    expert_indices_cpu = expert_indices_gpu.cpu().tolist()

    token_to_experts = {}
    for token_idx, expert_idx in zip(token_indices_cpu, expert_indices_cpu):
        if token_idx not in token_to_experts:
            token_to_experts[token_idx] = []
        token_to_experts[token_idx].append(expert_idx)

    for token_idx, expert_list in token_to_experts.items():
        global_token_id = token_offset + token_idx
        add_moe_stats_entry(
            statistics_collector, doc_id, layer_id, global_token_id, expert_list
        )


def _create_topk_gate_hook(layer_id):
    def _topk_gate_hook_inner(module, input, output):
        """Hook function for TopKGate to collect expert activation statistics."""
        global _moe_stats_data, _current_doc_id, _current_token_offset

        if _moe_stats_data is None or _current_doc_id is None:
            return

        # output from TopKGate is a tuple, combine_weights is the second element
        if isinstance(output, tuple) and len(output) > 1:
            combine_weights = output[1]
            _record_expert_stats(
                _moe_stats_data,
                _current_doc_id,
                layer_id,
                combine_weights,
                _current_token_offset,
            )

        return output

    return _topk_gate_hook_inner


def register_moe_hooks(model):
    """Register forward hooks on all TopKGate modules in the model."""
    hooks = []
    layer_pattern = re.compile(r"layers\.(\d+)\.")
    for name, module in model.named_modules():
        if hasattr(module, "__class__") and "TopKGate" in module.__class__.__name__:
            match = layer_pattern.search(name)
            if match:
                layer_id = int(match.group(1))
                hook_fn = _create_topk_gate_hook(layer_id)
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                # print(f"Registered MoE statistics hook on: {name} (layer {layer_id})")
            else:
                print(
                    f"Warning: Could not determine layer ID for TopKGate module: {name}. Hook not registered."
                )
    return hooks


def set_moe_statistics_context(stats_data, doc_id, token_offset=0):
    """Set global context for MoE statistics collection."""
    global _moe_stats_data, _current_doc_id, _current_token_offset
    _moe_stats_data = stats_data
    _current_doc_id = doc_id
    _current_token_offset = token_offset
    # print(f"[MoE Stats Debug] CONTEXT SET. doc_id: '{doc_id}', token_offset: {token_offset}", ranks=[0])


def clear_moe_statistics_context():
    """Clear global context for MoE statistics collection."""
    global _moe_stats_data, _current_doc_id, _current_token_offset
    _moe_stats_data = None
    _current_doc_id = None
    _current_token_offset = 0


# --- New Aggregate Statistics Functions ---

def _handle_batch_stats(stats_collector, layer_id, combine_weights):
    """
    Processes a batch of activations and updates the aggregate statistics collector.
    This function is designed to be called from a forward hook.
    """
    if combine_weights.dim() == 3:
        # Shape is (S, E, C), sum over capacity to get (S, E)
        active_expert_weights = torch.sum(combine_weights, dim=2)
    else:
        # Shape is already (S, E)
        active_expert_weights = combine_weights
    
    # Create a boolean mask of activated experts for each token
    active_mask = active_expert_weights > 1e-6

    # Sum across the token dimension to get counts per expert for this batch
    # Move to CPU to prevent GPU memory from accumulating across batches
    batch_counts = active_mask.sum(dim=0).cpu()

    num_experts = batch_counts.shape[0]

    # Initialize layer stats if not present
    if layer_id not in stats_collector:
        stats_collector[layer_id] = torch.zeros(num_experts, dtype=torch.float32)
    
    # Ensure tensors have same shape. This is a safeguard.
    if stats_collector[layer_id].shape[0] != num_experts:
        print(
            f"Warning: Expert count mismatch for layer {layer_id}. "
            f"Expected {stats_collector[layer_id].shape[0]}, got {num_experts}. "
            "Re-initializing stats for this layer."
        )
        stats_collector[layer_id] = torch.zeros(num_experts, dtype=torch.float32)

    # Add batch counts to the aggregate
    stats_collector[layer_id] += batch_counts.to(stats_collector[layer_id].dtype)


def _create_aggregate_gate_hook(stats_collector, layer_id):
    """
    Hook factory for aggregate statistics.
    Returns a hook function that collects expert activation counts per layer.
    """
    def _aggregate_hook_inner(module, input, output):
        # TopKGate output is a tuple, combine_weights is the second element
        if isinstance(output, tuple) and len(output) > 1:
            combine_weights = output[1]
            _handle_batch_stats(stats_collector, layer_id, combine_weights)
        return output
    return _aggregate_hook_inner


def register_aggregate_moe_hooks(model, stats_collector):
    """
    Registers forward hooks on all TopKGate modules for aggregate statistics.
    The stats_collector (an empty dict) will be populated by the hooks.
    """
    hooks = []
    layer_pattern = re.compile(r"layers\.(\d+)\.")
    for name, module in model.named_modules():
        if hasattr(module, "__class__") and "TopKGate" in module.__class__.__name__:
            match = layer_pattern.search(name)
            if match:
                layer_id = int(match.group(1))
                hook_fn = _create_aggregate_gate_hook(stats_collector, layer_id)
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
            else:
                print(
                    f"Warning: Could not determine layer ID for TopKGate module: {name}. Hook not registered."
                )
    return hooks


def save_aggregate_moe_stats(stats_collector, path):
    """
    Saves the aggregated statistics to a JSON file.
    Converts tensors to lists for serialization.
    """
    # Sort by layer ID for consistent output
    sorted_layers = sorted(stats_collector.keys())
    serializable_stats = {
        f"layer_{layer}": stats_collector[layer].tolist() for layer in sorted_layers
    }
    with open(path, "wt", encoding="utf--8") as fout:
        json.dump(serializable_stats, fout, ensure_ascii=False, indent=4)