from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal


mode_types = Literal['all', 'infer', 'eval', 'viz']


@dataclass
class GeneralArgs:
    # Path to the training configuration file
    config: Optional[str] = field(default=None, metadata={"help": "Train config file path"})
    # List of model names or paths
    models: Optional[List[str]] = field(default=None, metadata={"help": "List of model names or paths"})
    # List of dataset names or paths
    datasets: Optional[List[str]] = field(default=None, metadata={"help": "List of dataset names or paths"})
    # Path to the summarizer
    summarizer: Optional[str] = None
    # Enable debug mode
    debug: bool = field(default=False, metadata={"help": "Enable debug mode"})
    # Enable dry-run mode, where tasks are not actually executed
    dry_run: bool = field(default=False, metadata={"help": "Enable dry-run mode"})
    # Type of accelerator to use, e.g., 'vllm' or 'lmdeploy'
    accelerator: Literal['vllm', 'lmdeploy'] = field(default=None, metadata={"help": "Type of accelerator to use, e.g., 'vllm' or 'lmdeploy'"})
    # Running mode: 'all', 'infer', 'eval', or 'viz'
    mode: mode_types = field(default='all', metadata={"help": "Running mode: 'all', 'infer', 'eval', or 'viz'"})
    # Reuse previous outputs and results
    reuse: Optional[str] = field(default="latest", metadata={"help": "Reuse previous outputs and results"})
    # Directory where all outputs will be saved
    work_dir: Optional[str] = field(default=None, metadata={"help": "Directory where all outputs will be saved"})
    # Custom configuration directory
    config_dir: str = field(default='configs', metadata={"help": "Custom configuration directory"})
    # Report running status to Lark bot
    lark: bool = field(default=False, metadata={"help": "Report running status to Lark bot"})
    # Maximum number of workers to run in parallel
    max_num_workers: int = field(default=1, metadata={"help": "Maximum number of workers to run in parallel"})
    # Maximum number of tasks to run in parallel on one GPU
    max_workers_per_gpu: int = field(default=1, metadata={"help": "Maximum number of tasks to run in parallel on one GPU"})
    # Number of retries if a job fails
    retry: int = field(default=2, metadata={"help": "Number of retries if a job fails"})
    # Whether to dump evaluation details
    dump_eval_details: bool = field(default=False, metadata={"help": "Whether to dump evaluation details"})
    # Whether to dump extraction rate details
    dump_extract_rate: bool = field(default=False, metadata={"help": "Whether to dump extraction rate details"})

    # set srun args
    # slurm_parser: SlurmArgs = field(default_factory=SlurmArgs, metadata={"help": "Slurm arguments"})
    # set dlc args
    # dlc_parser: DlcArgs = field(default_factory=DlcArgs, metadata={"help": "DLC arguments"})

# @dataclass
# class SlurmArgs:
    slurm: bool = field(default=False, metadata={"help": "Enable Slurm mode"})
    # Slurm partition name
    partition: Optional[str] = field(default=None, metadata={"help": "Slurm partition name"})
    # Slurm quota type
    quotatype: Optional[str] = field(default=None, metadata={"help": "Slurm quota type"})
    # Slurm quality of service
    qos: Optional[str] = field(default=None, metadata={"help": "Slurm quality of service"})


# @dataclass
# class DlcArgs:
    dlc: bool = field(default=False, metadata={"help": "Enable DLC mode"})
    # Path to the Aliyun configuration file
    aliyun_cfg: str = field(default='~/.aliyun.cfg', metadata={"help": "Path to the Aliyun configuration file"})


# @dataclass
# class HfArgs:
    hf: bool = field(default=False, metadata={"help": "Enable HuggingFace mode"})
    # Type of the HuggingFace model, either 'base' or 'chat'
    hf_type: str = field(default='chat', metadata={"help": "Type of the HuggingFace model, either 'base' or 'chat'"})
    # Path to the HuggingFace model
    hf_path: Optional[str] = field(default=None, metadata={"help": "Path to the HuggingFace model"})
    # Additional keyword arguments for the HuggingFace model
    model_kwargs: Dict[str, str] = field(default_factory=dict, metadata={"help": "Additional keyword arguments for the HuggingFace model"})
    # Path to the HuggingFace tokenizer
    tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to the HuggingFace tokenizer"})
    # Additional keyword arguments for the tokenizer
    tokenizer_kwargs: Dict[str, str] = field(default_factory=dict, metadata={"help": "Additional keyword arguments for the tokenizer"})
    # Path to the PEFT model
    peft_path: Optional[str] = field(default=None, metadata={"help": "Path to the PEFT model"})
    # Additional keyword arguments for the PEFT model
    peft_kwargs: Dict[str, str] = field(default_factory=dict, metadata={"help": "Additional keyword arguments for the PEFT model"})
    # Additional keyword arguments for generation
    generation_kwargs: Dict[str, str] = field(default_factory=dict, metadata={"help": "Additional keyword arguments for generation"})
    # Maximum sequence length for the HuggingFace model
    max_seq_len: Optional[int] = None
    # Maximum output length for the HuggingFace model
    max_out_len: int = field(default=256, metadata={"help": "Maximum output length for the HuggingFace model"})
    # Minimum output length for the HuggingFace model
    min_out_len: int = field(default=1, metadata={"help": "Minimum output length for the HuggingFace model"})
    # Batch size for the HuggingFace model
    batch_size: int = field(default=8, metadata={"help": "Batch size for the HuggingFace model"})
    # Number of GPUS
    # num_gpus: int = field(default=1, metadata={"help": "Number of GPUS"})
    # Number of GPUs for the HuggingFace model
    hf_num_gpus: int = field(default=1, metadata={"help": "Number of GPUs for the HuggingFace model"})
    # Padding token ID for the HuggingFace model
    pad_token_id: Optional[int] = None
    # Stop words for the HuggingFace model
    stop_words: List[str] = field(default_factory=list, metadata={"help": "Stop words for the HuggingFace model"})

# @dataclass
# class CustomDatasetArgs:
    custom_dataset: bool = field(default=False, metadata={"help": "Enable custom dataset mode"})
    # Path to the custom dataset
    custom_dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to the custom dataset"})
    # Path to the custom dataset metadata
    custom_dataset_meta_path: Optional[str] = field(default=None, metadata={"help": "Path to the custom dataset metadata"})
    # Data type of the custom dataset, e.g., 'mcq' or 'qa'
    custom_dataset_data_type: Optional[Literal['mcq', 'qa']] = field(default=None, metadata={"help": "Data type of the custom dataset, e.g., 'mcq' or 'qa'"}) 
    # Inference method for the custom dataset, e.g., 'gen' or 'ppl'
    custom_dataset_infer_method: Optional[Literal['gen', 'ppl']] = field(default=None, metadata={"help": "Inference method for the custom dataset, e.g., 'gen' or 'ppl'"})


# @dataclass
# class Args:
#     general: GeneralArgs = field(default_factory=GeneralArgs, metadata={"help": "General arguments"})
#     slurm: SlurmArgs = field(default_factory=SlurmArgs, metadata={"help": "Slurm arguments"})
#     dlc: DlcArgs = field(default_factory=DlcArgs, metadata={"help": "DLC arguments"})
#     hf: HfArgs = field(default_factory=HfArgs, metadata={"help": "HuggingFace arguments"})
#     custom_dataset: CustomDatasetArgs = field(default_factory=CustomDatasetArgs, metadata={"help": "Custom dataset arguments"})

# def parse_args() -> Args:
#     # Implement the logic to parse command-line arguments and populate the dataclasses.
#     pass

# Example usage:
# args = parse_args()
# print(args.general.mode)

