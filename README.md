# LLM Fine-Tuner Unified

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A unified toolkit for fine-tuning and optimizing large language models (LLMs) across multiple frameworks and providers. Streamline your LLM workflow with a consistent interface for model training, evaluation, and deployment.

## Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
LLM Fine-Tuner Unified provides a standardized approach to fine-tuning large language models across different frameworks including Hugging Face Transformers, OpenAI, and custom model architectures. The project aims to simplify the process of adapting pre-trained models to specific tasks while maintaining reproducibility and scalability.

## Motivation
In the rapidly evolving field of AI, the ability to efficiently fine-tune and deploy LLMs is crucial. This project addresses the fragmentation in existing tools by providing:
- A unified interface for multiple model providers
- Reproducible training pipelines
- Scalable deployment options
- Comprehensive evaluation metrics
- Extensible architecture for custom integrations

## Features
- **Multi-Framework Support**: Compatible with Hugging Face, OpenAI, and custom models
- **Training Pipelines**: Configurable training loops with support for LoRA, QLoRA, and full fine-tuning
- **Evaluation Suite**: Comprehensive metrics including perplexity, BLEU, and custom evaluation functions
- **Hyperparameter Optimization**: Built-in support for Optuna and Ray Tune
- **Model Export**: Export to ONNX, TorchScript, or native formats
- **Experiment Tracking**: Integration with MLflow, Weights & Biases, and TensorBoard
- **Distributed Training**: Support for multi-GPU and distributed training

## Installation

### Prerequisites
- Python 3.8+
- pip
- CUDA 11.7+ (for GPU acceleration)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/sloppymo/llm-fine-tuner-unified.git
cd llm-fine-tuner-unified

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with pip
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Usage

### Basic Training Example
```python
from llm_fine_tuner import FineTuner, TrainingConfig

# Initialize with your model and dataset
trainer = FineTuner(
    model_name="gpt2",
    dataset_path="your_dataset.jsonl",
    output_dir="./output"
)

# Configure training
config = TrainingConfig(
    learning_rate=2e-5,
    batch_size=4,
    num_epochs=3,
    logging_steps=100
)

# Start training
trainer.train(config)
```

### Advanced Features
```python
# Hyperparameter optimization
from llm_fine_tuner.tuning import optimize_hyperparameters

best_params = optimize_hyperparameters(
    model_name="gpt2",
    dataset_path="your_dataset.jsonl",
    n_trials=20,
    direction="minimize"
)

# Model evaluation
results = trainer.evaluate(
    test_dataset="test_data.jsonl",
    metrics=["perplexity", "bleu"]
)
```

## Project Structure
```
llm-fine-tuner-unified/
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── docs/                  # Documentation
├── examples/              # Usage examples
├── llm_fine_tuner/        # Core package
│   ├── models/           # Model implementations
│   ├── training/         # Training loops
│   ├── evaluation/       # Evaluation metrics
│   ├── utils/            # Utility functions
│   └── __init__.py
├── tests/                 # Test suite
├── .gitignore
├── LICENSE
└── README.md
```

## Contributing
We welcome contributions from the community! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows our [Code of Conduct](CODE_OF_CONDUCT.md) and includes appropriate tests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, please open an issue or contact [Your Name] at [your.email@example.com].

---

<div align="center">
  <p>Built with ❤️ by <a href="https://github.com/sloppymo">Your Name</a></p>
  <p>🤖 Making AI more accessible and efficient</p>
</div>
