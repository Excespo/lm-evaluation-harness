# MoE Statistics Collection on Wikipedia NQ Open Categorized Data

This directory contains MoE statistics collection tasks for Wikipedia categorized training data from NQ Open dataset.

## Overview

Four specialized tasks have been created to collect MoE expert activation statistics on different Wikipedia domains:

1. **stats_pretrain_nq_open_geography** - Geography domain (~1.1MB)
2. **stats_pretrain_nq_open_history** - History domain (~4.2MB) 
3. **stats_pretrain_nq_open_literature** - Literature domain (~639KB)
4. **stats_pretrain_nq_open_politics** - Politics domain (~1.1MB)

## Data Format

Each task uses JSONL files with the following structure:
```json
{
  "id": 183,
  "text": "content from Wikipedia article...",
  "question": "related question",
  "answer": ["answer1", "answer2"]
}
```

The tasks focus on the `text` field which contains Wikipedia content for each domain.

## Dataset Configuration

All tasks use the standardized format:
```yaml
dataset_path: json
dataset_kwargs:
  data_files:
    train: /path/to/category/data.jsonl
```

This provides better flexibility and follows lm-evaluation-harness best practices.

## Usage

### Individual Tasks

```bash
# Run specific domain statistics
./run/run.sh --task moe_stat /path/to/model checkpoint stats_pretrain_nq_open_geography
./run/run.sh --task moe_stat /path/to/model checkpoint stats_pretrain_nq_open_history
./run/run.sh --task moe_stat /path/to/model checkpoint stats_pretrain_nq_open_literature
./run/run.sh --task moe_stat /path/to/model checkpoint stats_pretrain_nq_open_politics
```

### Using Helper Script

```bash
# Run single domain
./scripts/run_stats_pretrain_moe.sh /path/to/model checkpoint geography

# Run multiple domains
./scripts/run_stats_pretrain_moe.sh /path/to/model checkpoint geography history

# Run all domains
./scripts/run_stats_pretrain_moe.sh /path/to/model checkpoint all
```

### Mixed with Standard Tasks

```bash
# Combine with standard MoE tasks
./run/run.sh --task moe_stat /path/to/model checkpoint geography stats_pretrain_nq_open_geography
```

## Task Configuration

Each task is configured with:
- **Model**: HuggingFace with MoE patches
- **Batch Size**: 32
- **Generation**: Minimal (1 token) - focus on statistics, not generation quality
- **Target**: Empty string - no concern for model output
- **Metric**: exact_match (for compatibility, not the focus)
- **Task Type**: `moe_stat` only (not available in regular `moe` mode)

## Output

- Statistics files are saved in `outputs/{model_name}_{checkpoint_name}/hf/`
- Files with pattern `stats_moe_*.json` contain expert activation statistics
- Each domain generates separate statistics files for analysis

## Implementation Details

- Tasks are defined in `extra_moe_stats_tasks` array in `run/run.sh`
- Only available in `moe_stat` mode to avoid polluting regular MoE testing
- Uses real training sequences from each domain
- Designed for MoE models with expert activation tracking capabilities

## Data Paths

- Geography: `/aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/nq_open/categorized/Geography/Geography.jsonl`
- History: `/aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/nq_open/categorized/History/History.jsonl`
- Literature: `/aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/nq_open/categorized/Literature/Literature.jsonl`
- Politics: `/aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/nq_open/categorized/Politics/Politics.jsonl`

## Testing

Use the test script to verify task configuration:
```bash
./scripts/test_moe_stats_logic.sh
```

This will validate that all tasks are correctly configured and accessible only in `moe_stat` mode.

## Directory Structure

```
fpm_tasks/stats_pretrain_nq_open/
├── README.md                                    # This file
├── stats_pretrain_nq_open_geography.yaml       # Geography task config
├── stats_pretrain_nq_open_history.yaml         # History task config  
├── stats_pretrain_nq_open_literature.yaml      # Literature task config
└── stats_pretrain_nq_open_politics.yaml        # Politics task config
```

## Group Configuration

The main group configuration is defined in `fpm_tasks/_stats_pretrain_nq_open.yaml`, which includes all four domain tasks for convenient batch execution. 