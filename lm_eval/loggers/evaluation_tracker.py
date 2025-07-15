import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from datasets import load_dataset
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import (
    DatasetCard,
    DatasetCardData,
    HfApi,
    hf_hub_url,
)
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

from lm_eval.utils import (
    get_file_datetime,
    get_file_task_name,
    get_results_filenames,
    get_sample_results_filenames,
    handle_non_serializable,
    hash_string,
    sanitize_list,
    sanitize_model_name,
    sanitize_task_name,
)


eval_logger = logging.getLogger(__name__)


@dataclass(init=False)
class GeneralConfigTracker:
    """
    Tracker for the evaluation parameters.

    Attributes:
        model_source (str): Source of the model (e.g. Hugging Face, GGUF, etc.)
        model_name (str): Name of the model.
        model_name_sanitized (str): Sanitized model name for directory creation.
        start_time (float): Start time of the experiment. Logged at class init.
        end_time (float): Start time of the experiment. Logged when calling [`GeneralConfigTracker.log_end_time`]
        total_evaluation_time_seconds (str): Inferred total evaluation time in seconds (from the start and end times).
    """

    model_source: str = None
    model_name: str = None
    model_name_sanitized: str = None
    system_instruction: str = None
    system_instruction_sha: str = None
    fewshot_as_multiturn: bool = None
    chat_template: str = None
    chat_template_sha: str = None
    start_time: float = None
    end_time: float = None
    total_evaluation_time_seconds: str = None

    def __init__(self) -> None:
        """Starts the evaluation timer."""
        self.start_time = time.perf_counter()

    @staticmethod
    def _get_model_name(model_args: str) -> str:
        """Extracts the model name from the model arguments."""

        def extract_model_name(model_args: str, key: str) -> str:
            """Extracts the model name from the model arguments using a key."""
            args_after_key = model_args.split(key)[1]
            return args_after_key.split(",")[0]

        # order does matter, e.g. peft and delta are provided together with pretrained
        prefixes = ["peft=", "delta=", "pretrained=", "model=", "path=", "engine="]
        for prefix in prefixes:
            if prefix in model_args:
                return extract_model_name(model_args, prefix)
        return ""

    def log_experiment_args(
        self,
        model_source: str,
        model_args: str,
        system_instruction: str,
        chat_template: str,
        fewshot_as_multiturn: bool,
    ) -> None:
        """Logs model parameters and job ID."""
        self.model_source = model_source
        self.model_name = GeneralConfigTracker._get_model_name(model_args)
        self.model_name_sanitized = sanitize_model_name(self.model_name)
        self.system_instruction = system_instruction
        self.system_instruction_sha = (
            hash_string(system_instruction) if system_instruction else None
        )
        self.chat_template = chat_template
        self.chat_template_sha = hash_string(chat_template) if chat_template else None
        self.fewshot_as_multiturn = fewshot_as_multiturn

    def log_end_time(self) -> None:
        """Logs the end time of the evaluation and calculates the total evaluation time."""
        self.end_time = time.perf_counter()
        self.total_evaluation_time_seconds = str(self.end_time - self.start_time)


class EvaluationTracker:
    """
    Keeps track and saves relevant information of the evaluation process.
    Compiles the data from trackers and writes it to files, which can be published to the Hugging Face hub if requested.
    """

    def __init__(
        self,
        output_path: str = None,
        hub_results_org: str = "",
        hub_repo_name: str = "",
        details_repo_name: str = "",
        results_repo_name: str = "",
        push_results_to_hub: bool = False,
        push_samples_to_hub: bool = False,
        public_repo: bool = False,
        token: str = "",
        leaderboard_url: str = "",
        point_of_contact: str = "",
        gated: bool = False,
    ) -> None:
        """
        Creates all the necessary loggers for evaluation tracking.

        Args:
            output_path (str): Path to save the results. If not provided, the results won't be saved.
            hub_results_org (str): The Hugging Face organization to push the results to. If not provided, the results will be pushed to the owner of the Hugging Face token.
            hub_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will be pushed to `lm-eval-results`.
            details_repo_name (str): The name of the Hugging Face repository to push the details to. If not provided, the results will be pushed to `lm-eval-results`.
            result_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will not be pushed and will be found in the details_hub_repo.
            push_results_to_hub (bool): Whether to push the results to the Hugging Face hub.
            push_samples_to_hub (bool): Whether to push the samples to the Hugging Face hub.
            public_repo (bool): Whether to push the results to a public or private repository.
            token (str): Token to use when pushing to the Hugging Face hub. This token should have write access to `hub_results_org`.
            leaderboard_url (str): URL to the leaderboard on the Hugging Face hub on the dataset card.
            point_of_contact (str): Contact information on the Hugging Face hub dataset card.
            gated (bool): Whether to gate the repository.
        """
        self.general_config_tracker = GeneralConfigTracker()

        self.output_path = output_path
        self.push_results_to_hub = push_results_to_hub
        self.push_samples_to_hub = push_samples_to_hub
        self.public_repo = public_repo
        self.leaderboard_url = leaderboard_url
        self.point_of_contact = point_of_contact
        self.api = HfApi(token=token) if token else None
        self.gated_repo = gated

        if not self.api and (push_results_to_hub or push_samples_to_hub):
            raise ValueError(
                "Hugging Face token is not defined, but 'push_results_to_hub' or 'push_samples_to_hub' is set to True. "
                "Please provide a valid Hugging Face token by setting the HF_TOKEN environment variable."
            )

        if (
            self.api
            and hub_results_org == ""
            and (push_results_to_hub or push_samples_to_hub)
        ):
            hub_results_org = self.api.whoami()["name"]
            eval_logger.warning(
                f"hub_results_org was not specified. Results will be pushed to '{hub_results_org}'."
            )

        if hub_repo_name == "":
            details_repo_name = (
                details_repo_name if details_repo_name != "" else "lm-eval-results"
            )
            results_repo_name = (
                results_repo_name if results_repo_name != "" else details_repo_name
            )
        else:
            details_repo_name = hub_repo_name
            results_repo_name = hub_repo_name
            eval_logger.warning(
                "hub_repo_name was specified. Both details and results will be pushed to the same repository. Using hub_repo_name is no longer recommended, details_repo_name and results_repo_name should be used instead."
            )

        self.details_repo = f"{hub_results_org}/{details_repo_name}"
        self.details_repo_private = f"{hub_results_org}/{details_repo_name}-private"
        self.results_repo = f"{hub_results_org}/{results_repo_name}"
        self.results_repo_private = f"{hub_results_org}/{results_repo_name}-private"

    def save_results_aggregated(
        self,
        results: dict,
        samples: dict,
    ) -> None:
        """
        Saves the aggregated results and samples to the output path and pushes them to the Hugging Face hub if requested.

        Args:
            results (dict): The aggregated results to save.
            samples (dict): The samples results to save.
        """
        self.general_config_tracker.log_end_time()

        if self.output_path:
            try:
                eval_logger.info("Saving results aggregated")

                # calculate cumulative hash for each task - only if samples are provided
                task_hashes = {}
                if samples:
                    for task_name, task_samples in samples.items():
                        sample_hashes = [
                            s["doc_hash"] + s["prompt_hash"] + s["target_hash"]
                            for s in task_samples
                        ]
                        task_hashes[task_name] = hash_string("".join(sample_hashes))

                # update initial results dict
                results.update({"task_hashes": task_hashes})
                results.update(asdict(self.general_config_tracker))
                dumped = json.dumps(
                    results,
                    indent=2,
                    default=handle_non_serializable,
                    ensure_ascii=False,
                )

                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)

                self.date_id = datetime.now().isoformat().replace(":", "-")
                file_results_aggregated = path.joinpath(f"results_{self.date_id}.json")
                file_results_aggregated.open("w", encoding="utf-8").write(dumped)

                if self.api and self.push_results_to_hub:
                    repo_id = (
                        self.results_repo
                        if self.public_repo
                        else self.results_repo_private
                    )
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    self.api.upload_file(
                        repo_id=repo_id,
                        path_or_fileobj=str(
                            path.joinpath(f"results_{self.date_id}.json")
                        ),
                        path_in_repo=os.path.join(
                            self.general_config_tracker.model_name,
                            f"results_{self.date_id}.json",
                        ),
                        repo_type="dataset",
                        commit_message=f"Adding aggregated results for {self.general_config_tracker.model_name}",
                    )
                    eval_logger.info(
                        "Successfully pushed aggregated results to the Hugging Face Hub. "
                        f"You can find them at: {repo_id}"
                    )

            except Exception as e:
                eval_logger.warning("Could not save results aggregated")
                eval_logger.info(repr(e))
        else:
            eval_logger.info(
                "Output path not provided, skipping saving results aggregated"
            )

    def save_results_samples(
        self,
        task_name: str,
        samples: dict,
    ) -> None:
        """
        Saves the samples results to the output path and pushes them to the Hugging Face hub if requested.

        Args:
            task_name (str): The task name to save the samples for.
            samples (dict): The samples results to save.
        """
        if self.output_path:
            try:
                eval_logger.info(f"Saving per-sample results for: {task_name}")

                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)

                file_results_samples = path.joinpath(
                    f"samples_{task_name}_{self.date_id}.jsonl"
                )

                for sample in samples:
                    # we first need to sanitize arguments and resps
                    # otherwise we won't be able to load the dataset
                    # using the datasets library
                    arguments = {}
                    for i, arg in enumerate(sample["arguments"]):
                        arguments[f"gen_args_{i}"] = {}
                        for j, tmp in enumerate(arg):
                            arguments[f"gen_args_{i}"][f"arg_{j}"] = tmp

                    sample["resps"] = sanitize_list(sample["resps"])
                    sample["filtered_resps"] = sanitize_list(sample["filtered_resps"])
                    sample["arguments"] = arguments
                    sample["target"] = str(sample["target"])

                    sample_dump = (
                        json.dumps(
                            sample,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    with open(file_results_samples, "a", encoding="utf-8") as f:
                        f.write(sample_dump)

                if self.api and self.push_samples_to_hub:
                    repo_id = (
                        self.details_repo
                        if self.public_repo
                        else self.details_repo_private
                    )
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    try:
                        if self.gated_repo:
                            headers = build_hf_headers()
                            r = get_session().put(
                                url=f"https://huggingface.co/api/datasets/{repo_id}/settings",
                                headers=headers,
                                json={"gated": "auto"},
                            )
                            hf_raise_for_status(r)
                    except Exception as e:
                        eval_logger.warning("Could not gate the repository")
                        eval_logger.info(repr(e))
                    self.api.upload_folder(
                        repo_id=repo_id,
                        folder_path=str(path),
                        path_in_repo=self.general_config_tracker.model_name_sanitized,
                        repo_type="dataset",
                        commit_message=f"Adding samples results for {task_name} to {self.general_config_tracker.model_name}",
                    )
                    eval_logger.info(
                        f"Successfully pushed sample results for task: {task_name} to the Hugging Face Hub. "
                        f"You can find them at: {repo_id}"
                    )

            except Exception as e:
                eval_logger.warning("Could not save sample results")
                eval_logger.info(repr(e))
        else:
            eval_logger.info("Output path not provided, skipping saving sample results")

    def recreate_metadata_card(self) -> None:
        """
        Creates a metadata card for the evaluation results dataset and pushes it to the Hugging Face hub.
        """

        eval_logger.info("Recreating metadata card")
        repo_id = self.details_repo if self.public_repo else self.details_repo_private

        files_in_repo = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        results_files = get_results_filenames(files_in_repo)
        sample_files = get_sample_results_filenames(files_in_repo)

        # Build a dictionary to store the latest evaluation datetime for:
        # - Each tested model and its aggregated results
        # - Each task and sample results, if existing
        # i.e. {
        #     "org__model_name__gsm8k": "2021-09-01T12:00:00",
        #     "org__model_name__ifeval": "2021-09-01T12:00:00",
        #     "org__model_name__results": "2021-09-01T12:00:00"
        # }
        latest_task_results_datetime = defaultdict(lambda: datetime.min.isoformat())

        for file_path in sample_files:
            file_path = Path(file_path)
            filename = file_path.name
            model_name = file_path.parent
            task_name = get_file_task_name(filename)
            results_datetime = get_file_datetime(filename)
            task_name_sanitized = sanitize_task_name(task_name)
            # Results and sample results for the same model and task will have the same datetime
            samples_key = f"{model_name}__{task_name_sanitized}"
            results_key = f"{model_name}__results"
            latest_datetime = max(
                latest_task_results_datetime[samples_key],
                results_datetime,
            )
            latest_task_results_datetime[samples_key] = latest_datetime
            latest_task_results_datetime[results_key] = max(
                latest_task_results_datetime[results_key],
                latest_datetime,
            )

        # Create metadata card
        card_metadata = MetadataConfigs()

        # Add the latest aggregated results to the metadata card for easy access
        for file_path in results_files:
            file_path = Path(file_path)
            results_filename = file_path.name
            model_name = file_path.parent
            eval_date = get_file_datetime(results_filename)
            eval_date_sanitized = re.sub(r"[^\w\.]", "_", eval_date)
            results_filename = Path("**") / Path(results_filename).name
            config_name = f"{model_name}__results"
            sanitized_last_eval_date_results = re.sub(
                r"[^\w\.]", "_", latest_task_results_datetime[config_name]
            )

            if eval_date_sanitized == sanitized_last_eval_date_results:
                # Ensure that all results files are listed in the metadata card
                current_results = card_metadata.get(config_name, {"data_files": []})
                current_results["data_files"].append(
                    {"split": eval_date_sanitized, "path": [str(results_filename)]}
                )
                card_metadata[config_name] = current_results
                # If the results file is the newest, update the "latest" field in the metadata card
                card_metadata[config_name]["data_files"].append(
                    {"split": "latest", "path": [str(results_filename)]}
                )

        # Add the tasks details configs
        for file_path in sample_files:
            file_path = Path(file_path)
            filename = file_path.name
            model_name = file_path.parent
            task_name = get_file_task_name(filename)
            eval_date = get_file_datetime(filename)
            task_name_sanitized = sanitize_task_name(task_name)
            eval_date_sanitized = re.sub(r"[^\w\.]", "_", eval_date)
            results_filename = Path("**") / Path(filename).name
            config_name = f"{model_name}__{task_name_sanitized}"
            sanitized_last_eval_date_results = re.sub(
                r"[^\w\.]", "_", latest_task_results_datetime[config_name]
            )
            if eval_date_sanitized == sanitized_last_eval_date_results:
                # Ensure that all sample results files are listed in the metadata card
                current_details_for_task = card_metadata.get(
                    config_name, {"data_files": []}
                )
                current_details_for_task["data_files"].append(
                    {"split": eval_date_sanitized, "path": [str(results_filename)]}
                )
                card_metadata[config_name] = current_details_for_task
                # If the samples results file is the newest, update the "latest" field in the metadata card
                card_metadata[config_name]["data_files"].append(
                    {"split": "latest", "path": [str(results_filename)]}
                )

        # Get latest results and extract info to update metadata card examples
        latest_datetime = max(latest_task_results_datetime.values())
        latest_model_name = max(
            latest_task_results_datetime, key=lambda k: latest_task_results_datetime[k]
        )
        last_results_file = [
            f for f in results_files if latest_datetime.replace(":", "-") in f
        ][0]
        last_results_file_path = hf_hub_url(
            repo_id=repo_id, filename=last_results_file, repo_type="dataset"
        )
        latest_results_file = load_dataset(
            "json", data_files=last_results_file_path, split="train"
        )
        results_dict = latest_results_file["results"][0]
        new_dictionary = {"all": results_dict}
        new_dictionary.update(results_dict)
        results_string = json.dumps(new_dictionary, indent=4)

        dataset_summary = (
            "Dataset automatically created during the evaluation run of model "
        )
        if self.general_config_tracker.model_source == "hf":
            dataset_summary += f"[{self.general_config_tracker.model_name}](https://huggingface.co/{self.general_config_tracker.model_name})\n"
        else:
            dataset_summary += f"{self.general_config_tracker.model_name}\n"
        dataset_summary += (
            f"The dataset is composed of {len(card_metadata) - 1} configuration(s), each one corresponding to one of the evaluated task.\n\n"
            f"The dataset has been created from {len(results_files)} run(s). Each run can be found as a specific split in each "
            'configuration, the split being named using the timestamp of the run.The "train" split is always pointing to the latest results.\n\n'
            'An additional configuration "results" store all the aggregated results of the run.\n\n'
            "To load the details from a run, you can for instance do the following:\n"
        )
        if self.general_config_tracker.model_source == "hf":
            dataset_summary += (
                "```python\nfrom datasets import load_dataset\n"
                f'data = load_dataset(\n\t"{repo_id}",\n\tname="{latest_model_name}",\n\tsplit="latest"\n)\n```\n\n'
            )
        dataset_summary += (
            "## Latest results\n\n"
            f"These are the [latest results from run {latest_datetime}]({last_results_file_path.replace('/resolve/', '/blob/')}) "
            "(note that there might be results for other tasks in the repos if successive evals didn't cover the same tasks. "
            'You find each in the results and the "latest" split for each eval):\n\n'
            f"```python\n{results_string}\n```"
        )
        card_data = DatasetCardData(
            dataset_summary=dataset_summary,
            repo_url=f"https://huggingface.co/{self.general_config_tracker.model_name}",
            pretty_name=f"Evaluation run of {self.general_config_tracker.model_name}",
            leaderboard_url=self.leaderboard_url,
            point_of_contact=self.point_of_contact,
        )
        card_metadata.to_dataset_card_data(card_data)
        card = DatasetCard.from_template(
            card_data,
            pretty_name=card_data.pretty_name,
        )
        card.push_to_hub(repo_id, repo_type="dataset")

    def save_moe_statistics(
        self,
        per_task_moe_stats: dict,
        mapping_file: str = None,
    ) -> None:
        """
        Saves the MoE statistics to the output path and pushes them to the Hugging Face hub if requested.
        Also generates heatmap visualizations for each task.

        Args:
            per_task_moe_stats (dict): The MoE statistics to save, organized by task name.
            mapping_file (str): Optional path to expert mapping file for better labels.
        """
        if not self.output_path:
            eval_logger.info("Output path not provided, skipping saving MoE statistics")
            return

        eval_logger.info("Saving MoE statistics")

        path = Path(self.output_path if self.output_path else Path.cwd())
        path = path.joinpath(self.general_config_tracker.model_name_sanitized)
        path.mkdir(parents=True, exist_ok=True)

        # Use the same date_id as other results for consistency
        if not hasattr(self, 'date_id'):
            self.date_id = datetime.now().isoformat().replace(":", "-")
            
        file_moe_stats = path.joinpath(f"moe_stats_{self.date_id}.jsonl")

        with open(file_moe_stats, "w", encoding="utf-8") as f:
            for task_name, moe_stats in per_task_moe_stats.items():
                # Convert tensors to lists for JSON serialization
                serializable_stats = {}
                for layer, counts in moe_stats.items():
                    if hasattr(counts, 'tolist'):  # Check if it's a tensor
                        serializable_stats[layer] = counts.tolist()
                    else:
                        serializable_stats[layer] = counts

                record = {
                    "task": task_name, 
                    "stats": serializable_stats,
                    "timestamp": self.date_id
                }
                f.write(json.dumps(record, default=handle_non_serializable, ensure_ascii=False) + "\n")
                
                # Generate heatmap for this task
                eval_logger.info(f"Generating heatmap for task: {task_name}")
                self._create_task_heatmap(task_name, serializable_stats, str(path), mapping_file)

        eval_logger.info(f"MoE statistics saved to {file_moe_stats}")

        if self.api and self.push_results_to_hub:
            repo_id = (
                self.details_repo
                if self.public_repo
                else self.details_repo_private
            )
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=not self.public_repo,
                exist_ok=True,
            )
            self.api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=str(file_moe_stats),
                path_in_repo=os.path.join(
                    self.general_config_tracker.model_name_sanitized,
                    f"moe_stats_{self.date_id}.jsonl",
                ),
                repo_type="dataset",
                commit_message=f"Adding MoE statistics for {self.general_config_tracker.model_name}",
            )
            eval_logger.info(
                f"Successfully pushed MoE statistics to the Hugging Face Hub. "
                f"You can find them at: {repo_id}"
            )

    def _create_task_heatmap(self, task_name: str, moe_stats: dict, output_dir: str, mapping_file: str = None) -> None:
        """
        Create heat map for a single task showing all layers and experts
        
        Args:
            task_name (str): name of the task
            moe_stats (dict): MoE statistics for the task (layer_id: stats_list)
            output_dir (str): output directory for saving plots
            mapping_file (str): path to function_to_expert_indices mapping file
        """
        if not moe_stats:
            eval_logger.warning(f"No MoE statistics found for task: {task_name}")
            return
        
        # Read mapping file if provided
        mapping = {}
        if mapping_file and os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        
        # Convert MoE stats to a 2D matrix: layers x experts
        layer_names = list(moe_stats.keys())
        layer_names.sort()  # Sort layer names for consistent ordering
        
        if not layer_names:
            eval_logger.warning(f"No data found for task {task_name}")
            return
            
        # Get the number of experts from the first layer
        first_layer_data = moe_stats[layer_names[0]]
        n_experts = len(first_layer_data)
        n_layers = len(layer_names)
        
        # Create matrix: rows=layers, cols=experts
        stats_matrix = np.zeros((n_layers, n_experts))
        
        for i, layer_name in enumerate(layer_names):
            layer_data = moe_stats[layer_name]
            if len(layer_data) != n_experts:
                eval_logger.warning(f"Layer {layer_name} has {len(layer_data)} experts, expected {n_experts}")
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

        # Calculate average statistics
        avg_usage = np.mean(percentage_matrix, axis=0)
        
        expert_labels = self._create_expert_labels_from_mapping(mapping, n_experts)
        
        # Create figure with custom layout using gridspec
        fig = plt.figure(figsize=(max(12, n_experts * 0.8), max(10, n_layers + 4)))
        
        # Create gridspec with height ratios: main heatmap gets more space
        gs = fig.add_gridspec(3, 1, height_ratios=[n_layers, 1, 0.3], hspace=0.1)
        
        # 上图：详细的每层专家分布
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(percentage_matrix, cmap='viridis', aspect='auto', 
                        vmin=0, vmax=percentage_matrix.max())
        
        # 设置上图的标签和标题
        ax1.set_xticks(range(n_experts))
        ax1.set_xticklabels(expert_labels, rotation=45, ha='right')
        ax1.set_yticks(range(n_layers))
        ax1.set_yticklabels(layer_names)
        ax1.set_ylabel('Layers', fontsize=12)
        ax1.set_title(f'MoE Expert Usage Statistics - Task: {task_name}', fontsize=14, pad=20)
        
        # 在上图中添加数值标注
        for i in range(n_layers):
            for j in range(n_experts):
                text = ax1.text(j, i, f'{percentage_matrix[i, j]:.1f}',
                              ha="center", va="center", color="white" if percentage_matrix[i, j] > percentage_matrix.max()/2 else "black",
                              fontsize=8)
        
        # 下图：平均统计（单行）
        ax2 = fig.add_subplot(gs[1, 0])
        avg_matrix = avg_usage.reshape(1, -1)
        im2 = ax2.imshow(avg_matrix, cmap='viridis', aspect='auto', 
                        vmin=0, vmax=percentage_matrix.max())
        
        # 设置下图的标签 - 移除不必要的标签
        ax2.set_xticks(range(n_experts))
        ax2.set_xticklabels(expert_labels, rotation=45, ha='right')
        ax2.set_yticks([0])
        ax2.set_yticklabels(['Average'])
        
        # 在下图中添加数值标注
        for j in range(n_experts):
            text = ax2.text(j, 0, f'{avg_usage[j]:.1f}',
                          ha="center", va="center", color="white" if avg_usage[j] > percentage_matrix.max()/2 else "black",
                          fontsize=8)
        
        # 共享的颜色条
        cbar_ax = fig.add_subplot(gs[2, 0])
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Usage Percentage (%)', fontsize=12)
        
        # 确保上下图的x轴对齐
        ax1.set_xlim(-0.5, n_experts - 0.5)
        ax2.set_xlim(-0.5, n_experts - 0.5)
        
        # 移除上图的x轴标签（避免重复）
        ax1.set_xticklabels([])
        ax1.tick_params(axis='x', which='both', length=0)
        
        # 调整布局
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"moe_stats_{task_name}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        eval_logger.info(f"Saved MoE statistics heatmap for task '{task_name}' to: {filepath}")
        
        # Also save average statistics as a separate CSV file
        self._save_average_statistics(expert_labels, avg_usage, task_name, output_dir)

    def _create_expert_labels_from_mapping(self, mapping: dict, n_experts: int) -> list[str]:
        """
        Create expert labels from function_to_expert_indices mapping
        
        Args:
            mapping (dict): The mapping containing function_to_expert_indices
            n_experts (int): Total number of experts
            
        Returns:
            list[str]: List of expert labels
        """
        # Default labels
        expert_labels = [f"Expert_{i}" for i in range(n_experts)]
        
        # Check if mapping contains function_to_expert_indices
        if mapping and "function_to_expert_indices" in mapping:
            # mapping contains function_to_expert_indices key
            indices_mapping = mapping["function_to_expert_indices"]
            expert_to_function = {}
            for func_name, expert_indices in indices_mapping.items():
                for i, expert_idx in enumerate(expert_indices):
                    if expert_idx < n_experts:
                        expert_to_function[expert_idx] = f"{func_name}_{i}"
            
            # Update labels based on the mapping
            for expert_idx, label in expert_to_function.items():
                if expert_idx < len(expert_labels):
                    expert_labels[expert_idx] = label
        
        # If mapping is the function_to_expert_indices directly (not nested)
        elif mapping and all(isinstance(v, list) for v in mapping.values()):
            # mapping is directly function_to_expert_indices
            expert_to_function = {}
            for func_name, expert_indices in mapping.items():
                for i, expert_idx in enumerate(expert_indices):
                    if expert_idx < n_experts:
                        expert_to_function[expert_idx] = f"{func_name}_{i}"
            
            # Update labels based on the mapping
            for expert_idx, label in expert_to_function.items():
                if expert_idx < len(expert_labels):
                    expert_labels[expert_idx] = label
        
        return expert_labels

    def _save_average_statistics(self, expert_labels: list[str], avg_usage: np.ndarray, task_name: str, output_dir: str) -> None:
        """
        Save average expert usage statistics to a CSV file
        
        Args:
            expert_labels (list[str]): List of expert names
            avg_usage (np.ndarray): Average usage percentages
            task_name (str): Name of the task
            output_dir (str): Output directory
        """
        # Create DataFrame
        df = pd.DataFrame({
            'Expert': expert_labels,
            'Average_Usage_Percentage': avg_usage
        })
        
        # Sort by usage percentage (descending)
        df = df.sort_values('Average_Usage_Percentage', ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"moe_avg_stats_{task_name}_{timestamp}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        
        df.to_csv(csv_filepath, index=False, float_format='%.2f')
        eval_logger.info(f"Saved average MoE statistics to: {csv_filepath}")
