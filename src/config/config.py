"""Configuration management for RAG pipeline"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset"""
    name: str
    data_path: Path
    processed_path: Optional[Path] = None

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        if self.processed_path:
            self.processed_path = Path(self.processed_path)


@dataclass
class RAGConfig:
    """Main configuration for RAG pipeline"""
    # Dataset
    dataset: DatasetConfig

    # Paths
    project_root: Path = Path.cwd()
    index_dir: Path = None
    model_dir: Path = None

    # Embedding Model
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    use_finetuned: bool = False
    finetuned_model_path: Optional[Path] = None
    max_seq_length: int = 768

    # Retrieval
    top_k_retrieval: int = 50  # First stage: dense retrieval
    top_k_rerank: int = 10      # Second stage: reranking
    batch_size: int = 32

    # Reranker
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation (Azure OpenAI)
    azure_endpoint: str = ""
    azure_deployment: str = "gpt-4.1"
    azure_api_version: str = "2024-12-01-preview"
    azure_api_key: str = ""
    generation_temperature: float = 0.3
    generation_max_tokens: int = 2000

    # Fine-tuning
    finetune_epochs: int = 2
    finetune_batch_size: int = 8
    finetune_lr: float = 2e-5

    def __post_init__(self):
        self.project_root = Path(self.project_root)

        # Set default paths if not provided
        if self.index_dir is None:
            self.index_dir = self.project_root / "outputs" / "indexes" / self.dataset.name
        else:
            self.index_dir = Path(self.index_dir)

        if self.model_dir is None:
            self.model_dir = self.project_root / "models" / self.dataset.name
        else:
            self.model_dir = Path(self.model_dir)

        # Create directories
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        """Path to FAISS index file"""
        return self.index_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        """Path to document metadata file"""
        return self.index_dir / "doc_metadata.json"

    @property
    def config_cache_path(self) -> Path:
        """Path to config cache file"""
        return self.index_dir / "config.json"


def load_config(config_name: str, project_root: Optional[Path] = None) -> RAGConfig:
    """
    Load configuration from YAML file

    Args:
        config_name: Name of config file (without .yaml extension)
        project_root: Project root directory (default: current directory)

    Returns:
        RAGConfig instance
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    config_path = project_root / "experiments" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Build DatasetConfig
    dataset_dict = config_dict.pop('dataset')
    dataset_config = DatasetConfig(**dataset_dict)

    # Build RAGConfig
    config_dict['dataset'] = dataset_config
    config_dict['project_root'] = project_root

    return RAGConfig(**config_dict)


def save_config_template(output_path: Path, dataset_name: str = "hospital"):
    """
    Save a template configuration file

    Args:
        output_path: Path to save the template
        dataset_name: Name of the dataset
    """
    template = {
        'dataset': {
            'name': dataset_name,
            'data_path': f'data/raw/{dataset_name}/threads_with_summary.json',
            'processed_path': f'data/processed/{dataset_name}'
        },
        'embedding_model': 'Qwen/Qwen3-Embedding-0.6B',
        'use_finetuned': False,
        'finetuned_model_path': None,
        'max_seq_length': 768,
        'top_k_retrieval': 50,
        'top_k_rerank': 10,
        'batch_size': 32,
        'use_reranker': True,
        'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'azure_endpoint': 'https://your-endpoint.openai.azure.com/',
        'azure_deployment': 'gpt-4.1',
        'azure_api_version': '2024-12-01-preview',
        'azure_api_key': 'your-api-key-here',
        'generation_temperature': 0.3,
        'generation_max_tokens': 2000,
        'finetune_epochs': 2,
        'finetune_batch_size': 8,
        'finetune_lr': 2e-5
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True)

    print(f"Config template saved to: {output_path}")
