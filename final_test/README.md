---
dataset_info:
  config_name: default
  features:
  - name: expression
    dtype: string
  - name: evaluation
    dtype: string
  - name: tokens
    sequence: string
  - name: metadata
    dtype:
      token_count: int32
      operator_count: int32
      literal_count: int32
      nesting_depth: int32
      has_negation: bool
      is_error: bool
      complexity_score: float32
  splits:
  - name: train
    num_bytes: 9930
    num_examples: 30
  - name: validation
    num_bytes: 1962
    num_examples: 6
  - name: test
    num_bytes: 2544
    num_examples: 8
license: MIT
task_categories:
- text-classification
- logical-reasoning
language:
- en
size_categories:
- n<1K
---

# boloco-enhanced

Boolean Logic Expression Dataset with Enhanced Metadata

## Dataset Description

This is an enhanced version of the BoLoCo (Boolean Logic) dataset, featuring:

- **Rich Metadata**: Each example includes comprehensive metadata about token counts, operators, complexity scores, etc.
- **Modern Format**: JSON/JSONL format with full HuggingFace datasets compatibility
- **Reasoning Support**: Structured format ready for chain-of-thought reasoning tasks
- **Error Analysis**: Systematic error types and analysis capabilities

## Dataset Structure

### Data Fields

- `expression`: The Boolean logic expression as a string
- `evaluation`: The result ("T", "F", or "ERR")
- `tokens`: List of tokenized expression components
- `metadata`: Rich metadata including complexity metrics
- `reasoning_steps`: Optional step-by-step reasoning trace
- `error_type`: Classification of error type (if applicable)
- `created_at`: Creation timestamp

### Data Splits

- **train**: 30 examples
- **validation**: 6 examples
- **test**: 8 examples

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("boloco-enhanced")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"] 
test_data = dataset["test"]

# Example usage
for example in train_data:
    print(f"Expression: {example['expression']}")
    print(f"Result: {example['evaluation']}")
    print(f"Complexity: {example['metadata']['complexity_score']}")
```

## Dataset Creation

This dataset was generated using the BoLoCo toolkit with the following configuration:

```json
{
  "max_tokens": 5,
  "error_ratio": 0.05,
  "train_ratio": 0.7,
  "validate_ratio": 0.15,
  "test_ratio": 0.15,
  "seed": 42,
  "name": "boloco-enhanced",
  "version": "2.0.0",
  "description": "Boolean Logic Expression Dataset with Enhanced Metadata",
  "authors": [
    "BoLoCo Contributors"
  ],
  "license": "MIT"
}
```

## Statistics

### Train Split
- Total examples: 0
- True evaluations: 0
- False evaluations: 0
- Error evaluations: 0
- Average token count: 0.00
- Average complexity: 0.00

### Validation Split
- Total examples: 0
- True evaluations: 0
- False evaluations: 0
- Error evaluations: 0
- Average token count: 0.00
- Average complexity: 0.00

### Test Split
- Total examples: 0
- True evaluations: 0
- False evaluations: 0
- Error evaluations: 0
- Average token count: 0.00
- Average complexity: 0.00


## Citation

```bibtex
@misc{boloco-enhanced,
  title={BoLoCo Enhanced: Boolean Logic Expression Dataset},
  author={BoLoCo Contributors},
  year={2024},
  version={2.0.0},
  url={https://github.com/klusai/boloco}
}
```

## License

MIT
