# BoLoCo: Boolean Logic Expression Generator

## 🚀 **Version 2.0 - Enhanced with Modern AI/ML Integration**

BoLoCo is a **modern toolkit** for generating Boolean logic expression datasets with rich metadata, designed for training and evaluating logical reasoning capabilities in AI models. Version 2.0 introduces comprehensive HuggingFace integration, modern data formats, and enhanced metadata tracking.

## ✨ **New in Version 2.0**

- 🎯 **Modern Data Formats**: JSON/JSONL with rich metadata
- 🤗 **HuggingFace Integration**: Direct compatibility with `datasets` library
- 📊 **Enhanced Metadata**: Complexity scores, operator counts, nesting depth
- 🎨 **Rich CLI Experience**: Beautiful progress bars and formatted output
- 🔄 **Format Conversion**: Seamless legacy ↔ modern format conversion
- 📝 **Auto-Generated Dataset Cards**: Publication-ready documentation
- ✅ **Input Validation**: Comprehensive error checking and validation

## 📦 **Installation**

### Basic Installation
```bash
git clone https://github.com/klusai/boloco.git
cd boloco
pip install -e .
```

### Enhanced Installation (Recommended)
```bash
pip install -e ".[enhanced]"  # Includes HuggingFace + Rich CLI
```

### Full Installation (All Features)
```bash
pip install -e ".[full]"  # Includes transformers for advanced features
```

## 🚀 **Quick Start**

### Modern CLI (Recommended)
```bash
# Generate a modern dataset with rich metadata
boloco-modern generate --max-tokens 7 --output-dir ./data --format all

# Convert legacy format to modern
boloco-modern convert old_dataset.txt new_dataset.jsonl

# Validate dataset files
boloco-modern validate dataset.json
```

### Legacy CLI (Backward Compatible)
```bash
python -m boloco.boloco --mode generate --max_tokens 5 --error_ratio 0.05 --dir data
```

## 🎯 **Modern Dataset Format**

### Example JSON Structure
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
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Rich Metadata Features
- **Complexity Scoring**: Automated difficulty assessment
- **Operator Analysis**: Distribution and nesting patterns
- **Error Classification**: Systematic error type tracking
- **Provenance Tracking**: Complete generation history

## 🤗 **HuggingFace Integration**

### Direct Dataset Loading
```python
from datasets import load_dataset

# Load from local files
dataset = load_dataset("json", data_files="dataset.json")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"] 
test_data = dataset["test"]

# Iterate through examples
for example in train_data:
    print(f"Expression: {example['expression']}")
    print(f"Result: {example['evaluation']}")
    print(f"Complexity: {example['metadata']['complexity_score']}")
```

### Programmatic Generation
```python
from boloco.modern_formats import ModernBoLoCoDataset, ModernBoLoCoExample
from boloco.modern_cli import ModernBoLoCoGenerator

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
generator = ModernBoLoCoGenerator(config)
dataset = generator.generate_dataset()

# Convert to HuggingFace format
hf_dataset = dataset.to_huggingface_dataset()

# Save in multiple formats
dataset.save_json("dataset.json")
dataset.save_jsonl("dataset.jsonl") 
dataset.save_legacy_format("./legacy/")
dataset.create_dataset_card("README.md")
```

## 📊 **Advanced Features**

### Rich CLI Output
When you install with `[enhanced]`, you get:
- 🎨 **Colored Output**: Beautiful terminal formatting
- 📊 **Progress Bars**: Real-time generation progress
- 📋 **Statistical Tables**: Formatted dataset statistics
- ✅ **Status Indicators**: Clear success/error messages

### Format Conversion
```bash
# Legacy to modern
boloco-modern convert legacy_file.txt modern_file.jsonl --create-card

# Multiple format support
boloco-modern generate --format all  # Creates JSON, JSONL, HF, and legacy
```

### Validation & Analysis
```bash
# Validate dataset integrity
boloco-modern validate dataset.json --verbose

# Check legacy format
boloco-modern validate legacy_dataset.txt
```

## 🔧 **Configuration Options**

### Modern CLI Options
```bash
boloco-modern generate \
  --max-tokens 10 \           # Expression complexity
  --error-ratio 0.1 \         # Proportion of error examples
  --train-ratio 0.8 \         # Training split ratio
  --validate-ratio 0.1 \      # Validation split ratio
  --test-ratio 0.1 \          # Test split ratio
  --seed 42 \                 # Reproducibility seed
  --output-dir ./data \       # Output directory
  --format all \              # json|jsonl|hf|legacy|all
  --name "my-dataset" \       # Dataset name
  --version "1.0.0"           # Dataset version
```

### Legacy CLI Options (Backward Compatible)
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

### Modern Format Output
```
data/
├── dataset.json           # Complete dataset with metadata
├── dataset_train.jsonl    # Training split (JSONL)
├── dataset_validation.jsonl # Validation split
├── dataset_test.jsonl     # Test split
├── README.md             # Auto-generated dataset card
├── legacy/               # Legacy format files
│   ├── boloco-train-legacy.txt
│   ├── boloco-validation-legacy.txt
│   └── boloco-test-legacy.txt
└── hf_dataset/           # HuggingFace format
    ├── dataset_info.json
    ├── train/
    ├── validation/
    └── test/
```

### Legacy Format Output (Unchanged)
```
data/mt5/YYYYMMDD-HHMMSS/
├── boloco-train-mt5_se42_ex90_ra70.txt
├── boloco-validate-mt5_se42_ex18_ra15.txt
└── boloco-test-mt5_se42_ex22_ra15.txt
```

## 🎓 **Usage Examples**

### Research Workflow
```python
# Generate research dataset
from boloco.modern_cli import ModernBoLoCoGenerator

config = {
    "max_tokens": 15,
    "error_ratio": 0.2,
    "name": "research-logical-reasoning",
    "version": "1.0.0",
    "description": "Logical reasoning dataset for research"
}

generator = ModernBoLoCoGenerator(config)
dataset = generator.generate_dataset()

# Analyze complexity distribution
stats = dataset.metadata["statistics"]
print(f"Average complexity: {stats['train']['avg_complexity']:.2f}")

# Export for ML frameworks
hf_dataset = dataset.to_huggingface_dataset()
hf_dataset.push_to_hub("username/logical-reasoning")
```

### Model Training Pipeline
```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("json", data_files="dataset.json")

# Filter by complexity
complex_examples = dataset["train"].filter(
    lambda x: x["metadata"]["complexity_score"] > 10
)

# Prepare for training
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def prepare_examples(examples):
    inputs = [f"Evaluate: {expr}" for expr in examples["expression"]]
    targets = examples["evaluation"]
    return tokenizer(inputs, targets, truncation=True, padding=True)

tokenized_dataset = complex_examples.map(prepare_examples, batched=True)
```

## 🔗 **Integration Examples**

### With PyTorch
```python
from torch.utils.data import DataLoader
from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.json")
dataloader = DataLoader(dataset["train"], batch_size=32)

for batch in dataloader:
    expressions = batch["expression"]
    evaluations = batch["evaluation"]
    complexity = batch["metadata"]["complexity_score"]
    # Train your model...
```

### With scikit-learn
```python
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.json")
df = pd.DataFrame(dataset["train"])

# Feature extraction
X = df["metadata"].apply(pd.Series)  # Use metadata as features
y = df["evaluation"]  # Target labels

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)
```

## 🤝 **Contributing**

We welcome contributions! The modern codebase is designed for extensibility:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** your enhancements
4. **Test** with both legacy and modern formats
5. **Submit** a pull request

### Development Setup
```bash
git clone https://github.com/klusai/boloco.git
cd boloco
pip install -e ".[dev]"  # Includes all dev dependencies
```

## 📈 **Performance & Compatibility**

### Backward Compatibility
- ✅ **Legacy CLI**: Fully functional
- ✅ **Legacy Format**: Supported in all tools
- ✅ **Old Scripts**: Work without modification

### Performance Improvements
- 🚀 **Faster Generation**: Optimized algorithms
- 📊 **Rich Metadata**: Computed efficiently
- 💾 **Multiple Formats**: Parallel export options

## 📚 **Documentation**

- **Modern API**: See `boloco/modern_formats.py`
- **CLI Reference**: `boloco-modern --help`
- **Legacy Docs**: Original README sections below
- **Examples**: `/examples` directory (coming soon)

## 📄 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🔄 **Migration Guide**

### From Legacy to Modern

**Old way:**
```bash
python boloco.py --mode generate --max_tokens 5 --dir data
```

**New way:**
```bash
boloco-modern generate --max-tokens 5 --output-dir data --format all
```

### Converting Existing Datasets
```bash
# Convert all legacy files in a directory
find ./old_data -name "*.txt" -exec boloco-modern convert {} {}.jsonl \;

# Batch convert with dataset cards
boloco-modern convert legacy_dataset.txt modern_dataset.jsonl --create-card
```

---

# 📖 **Legacy Documentation**

*The following sections preserve the original BoLoCo documentation for reference and backward compatibility.*

## Description
BoLoCo is a tool designed to create datasets consisting of Boolean logic expressions. The script generates valid and erroneous expressions, which are useful for training and evaluating logic-based machine learning models. It organizes the data into train, validate, and test sets with flexible configurations for expression complexity and error ratios.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/klusai/boloco.git
   cd boloco
   ```

2. **Setup Environment**:
   Ensure you have Python 3 and the necessary packages installed. Recommended to set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Generate a Dataset
To generate a dataset with specified parameters, run:
```bash
python boloco.py --mode generate --max_tokens 5 --error_ratio 0.05 --dir data
```

- `--max_tokens`: Maximum number of tokens in each logic expression (default is 5).
- `--error_ratio`: The proportion of errors to introduce in the dataset (default is 0.05).
- `--dir`: Directory path to save the generated datasets.

**Example Output:**
```plaintext
<s> ( F ) <eval/> F </s>
<s> ( T OR F ) <eval/> T </s>
<s> AND F AND F <eval/> <err/> </s>
```

### Collect Dataset Statistics
To compute and print statistics of existing datasets, use:
```bash
python boloco.py --mode stats --dir data
```
- `--dir`: Directory path where the dataset files are stored.

## Configuration Options
- `--train_ratio` (default: 0.7): Ratio of data to allocate for training.
- `--validate_ratio` (default: 0.15): Ratio of data to allocate for validation.
- `--test_ratio` (default: 0.15): Ratio of data to allocate for testing.
- `--seed` (default: 42): Seed for random number generators to ensure reproducibility.

## Output Files
Upon execution, the script generates files in the specified directory with a structure such as `data/mt5/YYYYMMDD-HHMMSS`, containing:
- `boloco-train-mt5_se42_ex90_ra70.txt`
- `boloco-validate-mt5_se42_ex18_ra15.txt`
- `boloco-test-mt5_se42_ex22_ra15.txt`

Each file contains lines formatted as:
```
<s> ( Expression ) <eval/> Evaluation_Result </s>
```