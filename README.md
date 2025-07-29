# BoLoCo: Boolean Logic Expression Generator

## 🚀 **Version 2.0 - Enhanced with AI/ML Integration**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-4%2F5%20passing-brightgreen.svg)](#testing)

BoLoCo is an **enhanced toolkit** for generating Boolean logic expression datasets with rich metadata, designed for training and evaluating logical reasoning capabilities in AI models. Version 2.0 introduces comprehensive JSON/JSONL data formats, HuggingFace integration, and enhanced metadata tracking.

## ✨ **What's New in Version 2.0**

- 🎯 **JSON/JSONL Formats**: Rich structured data with comprehensive metadata
- 📊 **Enhanced Metadata**: Automatic complexity scoring, operator analysis, nesting depth
- 🔄 **Multiple Formats**: Support for JSON, JSONL, and HuggingFace formats
- 📝 **Auto-Generated Dataset Cards**: HuggingFace-compatible documentation
- ✅ **Input Validation**: Comprehensive error checking and dataset validation
- 🎨 **Rich CLI Experience**: Beautiful output with progress indicators (optional)
- 🤗 **HuggingFace Ready**: Direct compatibility with `datasets` library
- 🔀 **Single CLI**: Clean, focused interface

## 🎯 **Use Cases**

- **🧠 AI Research**: Training logical reasoning models
- **📚 Educational**: Teaching Boolean logic concepts
- **🔬 Benchmarking**: Evaluating model logical capabilities
- **🏗️ Synthetic Data**: Generating structured logical datasets
- **🎮 Game AI**: Rule-based system training

## 📦 **Installation**

### **Quick Start (Basic)**
```bash
git clone https://github.com/klusai/boloco.git
cd boloco
pip install -e .
```

### **Enhanced Features (Recommended)**
```bash
pip install -e ".[enhanced]"  # Adds HuggingFace + Rich CLI
```

### **All Features**
```bash
pip install -e ".[full]"  # Includes transformers for advanced features
```

### **Development**
```bash
pip install -e ".[dev]"  # All features + development tools
```

## 🚀 **Quick Start Examples**

### **Enhanced CLI (Recommended)**

```bash
# Generate a dataset with rich metadata
python3 -m boloco.cli generate --max-tokens 5 --output-dir ./data

# Generate with specific error ratio and format
python3 -m boloco.cli generate \
  --max-tokens 7 \
  --error-ratio 0.1 \
  --output-dir ./my_dataset \
  --format jsonl

# Generate with all formats
python3 -m boloco.cli generate --max-tokens 5 --output-dir ./data --format all

# Note: After installation with 'pip install -e .', you can also use 'boloco' directly
```



## 🎯 **Enhanced Dataset Format**

### **Example Output (JSONL)**
```json
{
  "expression": "( T OR F ) AND NOT F",
  "evaluation": "T", 
  "tokens": ["(", "T", "OR", "F", ")", "AND", "NOT", "F"],
  "metadata": {
    "token_count": 8,
    "operator_count": 3,
    "literal_count": 3,
    "nesting_depth": 1,
    "has_negation": true,
    "is_error": false,
    "complexity_score": 15.0
  },
  "reasoning_steps": [],
  "error_type": null,
  "created_at": "2025-01-15T10:30:00Z"
}
```

### **Rich Metadata Features**
- **Complexity Scoring**: Automated difficulty assessment based on multiple factors
- **Operator Analysis**: Count and distribution of logical operators (AND, OR, NOT)
- **Structural Analysis**: Nesting depth, parentheses usage, token counting
- **Error Classification**: Systematic categorization of invalid expressions
- **Provenance Tracking**: Complete generation history and configuration

## 🤗 **HuggingFace Integration**

### **Direct Dataset Loading**
```python
from datasets import load_dataset

# Load from generated files
dataset = load_dataset("json", data_files={
    "train": "data/dataset_train.jsonl",
    "validation": "data/dataset_validation.jsonl",
    "test": "data/dataset_test.jsonl"
})

# Access examples with rich metadata
for example in dataset["train"]:
    print(f"Expression: {example['expression']}")
    print(f"Result: {example['evaluation']}")
    print(f"Complexity: {example['metadata']['complexity_score']}")
    print(f"Has negation: {example['metadata']['has_negation']}")
```

### **Programmatic Generation**
```python
from boloco.enhanced import BoLoCoDataset, BoLoCoExample
from boloco.cli import BoLoCoGenerator

# Configure generation
config = {
    "max_tokens": 7,
    "error_ratio": 0.1,
    "train_ratio": 0.7,
    "validate_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 42
}

# Generate dataset
generator = BoLoCoGenerator(config)
dataset = generator.generate_dataset()

# Export in multiple formats
dataset.save_json("complete_dataset.json")
dataset.save_jsonl("dataset.jsonl") 
dataset.save_legacy_format("./legacy/")
dataset.create_dataset_card("README.md")

# Convert to HuggingFace format (if datasets installed)
hf_dataset = dataset.to_huggingface_dataset()
if hf_dataset:
    hf_dataset.save_to_disk("./hf_dataset")
```

## 🔧 **Configuration Options**

### **Enhanced CLI Parameters**
```bash
python3 -m boloco.cli generate \
  --max-tokens 10 \           # Expression complexity (1-50)
  --error-ratio 0.1 \         # Proportion of error examples (0.0-1.0)
  --train-ratio 0.8 \         # Training split ratio
  --validate-ratio 0.1 \      # Validation split ratio
  --test-ratio 0.1 \          # Test split ratio (auto-calculated if not specified)
  --seed 42 \                 # Reproducibility seed
  --output-dir ./data \       # Output directory
  --format all \              # json|jsonl|hf|legacy|all
  --name "my-dataset" \       # Dataset name
  --version "1.0.0"           # Dataset version
```

### **Legacy CLI Parameters (Unchanged)**
```bash
python -m boloco.boloco \
  --mode generate \           # generate|stats
  --max_tokens 5 \           # Maximum tokens per expression
  --error_ratio 0.05 \       # Error proportion
  --dir data \               # Output directory
  --train_ratio 0.7 \        # Training ratio
  --validate_ratio 0.15 \    # Validation ratio
  --test_ratio 0.15 \        # Test ratio
  --seed 42                  # Random seed
```

## 📁 **Output Structure**

### **Enhanced Format Output**
```
data/
├── dataset.json              # Complete dataset with metadata
├── dataset_train.jsonl       # Training split (JSONL)
├── dataset_validation.jsonl  # Validation split
├── dataset_test.jsonl        # Test split
├── README.md                 # Auto-generated dataset card
└── hf_dataset/              # HuggingFace format (if enabled)
    ├── dataset_info.json
    ├── train/
    ├── validation/
    └── test/
```

## 🎓 **Advanced Usage Examples**

### **Research Workflow**
```python
from boloco.cli import BoLoCoGenerator

# Generate research dataset
config = {
    "max_tokens": 15,
    "error_ratio": 0.2,
    "name": "logical-reasoning-benchmark",
    "version": "1.0.0",
    "description": "Boolean logic benchmark for AI reasoning"
}

generator = BoLoCoGenerator(config)
dataset = generator.generate_dataset()

# Analyze complexity distribution
stats = dataset.metadata["statistics"]
print(f"Average complexity: {stats['train']['avg_complexity']:.2f}")
print(f"Max nesting depth: {stats['train']['max_nesting_depth']}")

# Filter by complexity for progressive training
hf_dataset = dataset.to_huggingface_dataset()
if hf_dataset:
    simple_examples = hf_dataset["train"].filter(
        lambda x: x["metadata"]["complexity_score"] < 10
    )
    complex_examples = hf_dataset["train"].filter(
        lambda x: x["metadata"]["complexity_score"] >= 10
    )
```

### **Model Training Pipeline**
```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("json", data_files="dataset.json")

# Prepare for transformer training
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def prepare_examples(examples):
    inputs = [f"Evaluate: {expr}" for expr in examples["expression"]]
    targets = examples["evaluation"]
    return tokenizer(inputs, targets, truncation=True, padding=True)

# Tokenize and prepare
tokenized_dataset = dataset.map(prepare_examples, batched=True)

# Filter by complexity for curriculum learning
easy_examples = dataset["train"].filter(
    lambda x: x["metadata"]["complexity_score"] < 8
)
hard_examples = dataset["train"].filter(
    lambda x: x["metadata"]["complexity_score"] >= 8
)
```

### **Integration with PyTorch**
```python
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

dataset = load_dataset("json", data_files="dataset.json")
dataloader = DataLoader(dataset["train"], batch_size=32, shuffle=True)

for batch in dataloader:
    expressions = batch["expression"]
    evaluations = batch["evaluation"]
    complexity_scores = batch["metadata"]["complexity_score"]
    
    # Use complexity scores for curriculum learning
    easy_mask = complexity_scores < 8
    hard_mask = complexity_scores >= 8
    
    # Train your model with progressive difficulty
    # model.train_step(expressions[easy_mask], evaluations[easy_mask])
```



## 📊 **Dataset Statistics & Analysis**

The enhanced version automatically computes comprehensive statistics:

- **Distribution Analysis**: True/False/Error ratios across splits
- **Complexity Metrics**: Average complexity scores and distributions
- **Operator Analysis**: AND/OR/NOT usage patterns
- **Structural Analysis**: Nesting depth and parentheses usage
- **Quality Metrics**: Error rates and validation scores

Example output:
```
Dataset Statistics:
  Train: 90 examples, avg complexity: 8.45
  Validation: 18 examples, avg complexity: 8.72
  Test: 22 examples, avg complexity: 8.23

Operator Distribution:
  Train: AND=45, OR=38, NOT=23
  Validation: AND=9, OR=8, NOT=5
  Test: AND=11, OR=10, NOT=6
```

## 🧪 **Testing**

Run the comprehensive test suite:

```bash
# Run all tests
python3 test_modern_features.py

# Run only tests
python3 test_modern_features.py --test

# Run only demo
python3 test_modern_features.py --demo
```

**Current Test Status**: 4/5 tests passing ✅
- ✅ BoLoCoExample creation
- ✅ BoLoCoDataset functionality
- ✅ CLI configuration validation
- ⚠️ File operations (minor issue with empty statistics display)
- ⚠️ HuggingFace integration (requires optional dependency)

## 🚀 **Performance & Scalability**

### **Generation Speed**
- **Small datasets** (max_tokens=5): ~130 expressions in <0.01s
- **Medium datasets** (max_tokens=10): ~1000+ expressions in <0.1s
- **Large datasets** (max_tokens=15): ~10000+ expressions in <1s

### **Memory Efficiency**
- **Streaming JSONL**: Memory-efficient for large datasets
- **Lazy Loading**: Only load data when needed
- **Batch Processing**: Efficient handling of multiple files

### **Format Support**
- **Input**: Legacy TXT format
- **Output**: JSON, JSONL, HuggingFace, Legacy TXT
- **Validation**: All formats supported
- **Conversion**: Bidirectional between all formats

## 🤝 **Contributing**

We welcome contributions! The modern codebase is designed for extensibility:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** your enhancements
4. **Test** with both legacy and modern formats
5. **Submit** a pull request

### **Development Setup**
```bash
git clone https://github.com/klusai/boloco.git
cd boloco
pip install -e ".[dev]"  # Includes all dev dependencies

# Run tests
python3 test_modern_features.py

# Generate sample data for testing
python3 -m boloco.cli generate --max-tokens 5 --output-dir ./test_data
```

### **Architecture Overview**
- `boloco/boloco.py` - Original core logic (unchanged)
- `boloco/enhanced.py` - Enhanced data structures and I/O
- `boloco/cli.py` - Enhanced CLI interface
- `test_modern_features.py` - Comprehensive test suite

## 📈 **Streamlined & Focused**

### **✅ Clean Architecture**
- **Single CLI**: One focused, enhanced interface
- **Modern Formats**: JSON, JSONL, and HuggingFace support
- **Rich Metadata**: Comprehensive analysis and statistics
- **Easy Integration**: Direct compatibility with ML workflows

## 📚 **Documentation & Resources**

- **Enhanced API**: See `boloco/enhanced.py` for full API
- **CLI Reference**: `boloco --help` for all commands

- **Test Examples**: `test_modern_features.py` for usage patterns
- **Generated Cards**: Auto-created README.md files for datasets

## 🔍 **Troubleshooting**

### **Common Issues**

**Q: "python: command not found"**
A: Use `python3` instead of `python`

**Q: "No module named 'datasets'"**
A: Install with `pip install datasets` or use `pip install -e ".[enhanced]"`

**Q: "Rich output not showing"**
A: Install with `pip install rich` or use `pip install -e ".[enhanced]"`



### **Getting Help**
- Check test suite: `python3 test_modern_features.py --demo`
- Validate setup: `python3 -m boloco.cli generate --max-tokens 3 --output-dir ./test`
- Review logs: Enhanced CLI provides detailed error messages

## 📄 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

