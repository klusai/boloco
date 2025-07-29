"""
Enhanced data formats and HuggingFace integration for BoLoCo.

This module provides JSON/JSONL formats, comprehensive metadata tracking,
and seamless integration with HuggingFace datasets ecosystem.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

# Optional imports for HuggingFace integration
try:
    from datasets import Dataset, DatasetDict, Features, Value, Sequence
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    Dataset = DatasetDict = Features = Value = Sequence = None

logger = logging.getLogger("boloco.enhanced")


class BoLoCoExample:
    """
    Enhanced representation of a Boolean logic expression example with rich metadata.
    """
    
    def __init__(
        self,
        expression: str,
        evaluation: str,
        tokens: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        reasoning_steps: Optional[List[Dict[str, Any]]] = None,
        error_type: Optional[str] = None
    ):
        self.expression = expression
        self.evaluation = evaluation  # "T", "F", or "ERR"
        self.tokens = tokens
        self.metadata = metadata or {}
        self.reasoning_steps = reasoning_steps or []
        self.error_type = error_type
        self.created_at = datetime.utcnow().isoformat()
        
        # Auto-compute metadata
        self._compute_metadata()
    
    def _compute_metadata(self):
        """Automatically compute metadata from the expression."""
        # Compute individual metrics first
        token_count = len(self.tokens)
        operator_count = self._count_operators()
        literal_count = self._count_literals()
        nesting_depth = self._compute_nesting_depth()
        has_negation = "NOT" in self.tokens
        is_error = self.evaluation == "ERR"
        
        # Update metadata with computed values
        self.metadata.update({
            "token_count": token_count,
            "operator_count": operator_count,
            "literal_count": literal_count,
            "nesting_depth": nesting_depth,
            "has_negation": has_negation,
            "is_error": is_error,
            "complexity_score": self._compute_complexity()
        })
    
    def _count_operators(self) -> int:
        """Count logical operators in the expression."""
        operators = ["AND", "OR", "NOT"]
        return sum(1 for token in self.tokens if token in operators)
    
    def _count_literals(self) -> int:
        """Count literals (T, F) in the expression."""
        return sum(1 for token in self.tokens if token in ["T", "F"])
    
    def _compute_nesting_depth(self) -> int:
        """Compute maximum parentheses nesting depth."""
        depth = 0
        max_depth = 0
        for token in self.tokens:
            if token == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif token == ")":
                depth -= 1
        return max_depth
    
    def _compute_complexity(self) -> float:
        """Compute a complexity score based on various factors."""
        # Use direct computation if metadata not yet populated
        token_count = self.metadata.get("token_count", len(self.tokens))
        operator_count = self.metadata.get("operator_count", self._count_operators())
        nesting_depth = self.metadata.get("nesting_depth", self._compute_nesting_depth())
        has_negation = self.metadata.get("has_negation", "NOT" in self.tokens)
        
        base_score = token_count
        operator_weight = operator_count * 2
        nesting_weight = nesting_depth * 3
        negation_weight = 2 if has_negation else 0
        
        return base_score + operator_weight + nesting_weight + negation_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format suitable for JSON serialization."""
        return {
            "expression": self.expression,
            "evaluation": self.evaluation,
            "tokens": self.tokens,
            "metadata": self.metadata,
            "reasoning_steps": self.reasoning_steps,
            "error_type": self.error_type,
            "created_at": self.created_at
        }
    
    def to_legacy_format(self) -> str:
        """Convert to legacy BoLoCo format for backward compatibility."""
        return f"<s> {self.expression} <eval/> {self.evaluation} </s>"
    
    @classmethod
    def from_legacy_format(cls, legacy_line: str) -> "BoLoCoExample":
        """Create from legacy BoLoCo format."""
        # Parse legacy format: "<s> expression <eval/> result </s>"
        parts = legacy_line.strip().split()
        if len(parts) < 4 or parts[0] != "<s>" or parts[-1] != "</s>":
            raise ValueError(f"Invalid legacy format: {legacy_line}")
        
        eval_idx = parts.index("<eval/>")
        expression_tokens = parts[1:eval_idx]
        expression = " ".join(expression_tokens)
        evaluation = parts[eval_idx + 1]
        
        return cls(
            expression=expression,
            evaluation=evaluation,
            tokens=expression_tokens
        )


class BoLoCoDataset:
    """
    Enhanced dataset container with comprehensive metadata and format support.
    """
    
    def __init__(
        self,
        name: str = "boloco-enhanced",
        version: str = "2.0.0",
        description: str = "Boolean Logic Expression Dataset with Enhanced Metadata",
        authors: List[str] = None,
        license: str = "MIT"
    ):
        self.name = name
        self.version = version
        self.description = description
        self.authors = authors or ["BoLoCo Contributors"]
        self.license = license
        self.created_at = datetime.utcnow().isoformat()
        
        self.splits = {
            "train": [],
            "validation": [],
            "test": []
        }
        
        self.metadata = {
            "format_version": "2.0",
            "generation_config": {},
            "statistics": {},
            "provenance": {
                "created_at": self.created_at,
                "tool": "boloco-enhanced",
                "version": version
            }
        }
    
    def add_example(self, example: BoLoCoExample, split: str = "train"):
        """Add an example to the specified split."""
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.splits.keys())}")
        
        self.splits[split].append(example)
    
    def add_examples(self, examples: List[BoLoCoExample], split: str = "train"):
        """Add multiple examples to the specified split."""
        for example in examples:
            self.add_example(example, split)
    
    def update_metadata(self, config: Dict[str, Any]):
        """Update dataset metadata with generation configuration."""
        self.metadata["generation_config"].update(config)
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        stats = {}
        
        for split_name, examples in self.splits.items():
            split_stats = {
                "total_examples": len(examples),
                "true_evaluations": sum(1 for ex in examples if ex.evaluation == "T"),
                "false_evaluations": sum(1 for ex in examples if ex.evaluation == "F"),
                "error_evaluations": sum(1 for ex in examples if ex.evaluation == "ERR"),
                "avg_token_count": sum(ex.metadata["token_count"] for ex in examples) / len(examples) if examples else 0,
                "avg_complexity": sum(ex.metadata["complexity_score"] for ex in examples) / len(examples) if examples else 0,
                "max_nesting_depth": max((ex.metadata["nesting_depth"] for ex in examples), default=0),
                "operator_distribution": self._compute_operator_distribution(examples)
            }
            stats[split_name] = split_stats
        
        self.metadata["statistics"] = stats
    
    def _compute_operator_distribution(self, examples: List[BoLoCoExample]) -> Dict[str, int]:
        """Compute distribution of operators across examples."""
        operators = ["AND", "OR", "NOT"]
        distribution = {op: 0 for op in operators}
        
        for example in examples:
            for token in example.tokens:
                if token in operators:
                    distribution[token] += 1
        
        return distribution
    
    def save_jsonl(self, output_path: Union[str, Path], split: Optional[str] = None):
        """Save dataset in JSONL format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        splits_to_save = [split] if split else self.splits.keys()
        
        for split_name in splits_to_save:
            if split:
                file_path = output_path
            else:
                file_path = output_path.parent / f"{output_path.stem}_{split_name}.jsonl"
            
            with open(file_path, "w", encoding="utf-8") as f:
                for example in self.splits[split_name]:
                    json.dump(example.to_dict(), f, ensure_ascii=False)
                    f.write("\n")
            
            logger.info(f"Saved {len(self.splits[split_name])} examples to {file_path}")
    
    def save_json(self, output_path: Union[str, Path]):
        """Save complete dataset with metadata in JSON format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset_dict = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "authors": self.authors,
            "license": self.license,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "splits": {
                split_name: [ex.to_dict() for ex in examples]
                for split_name, examples in self.splits.items()
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved complete dataset to {output_path}")
    
    def save_legacy_format(self, output_dir: Union[str, Path]):
        """Save in legacy BoLoCo format for backward compatibility."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, examples in self.splits.items():
            if not examples:
                continue
                
            file_path = output_dir / f"boloco-{split_name}-legacy.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                for example in examples:
                    f.write(example.to_legacy_format() + "\n")
            
            logger.info(f"Saved {len(examples)} examples in legacy format to {file_path}")
    
    def to_huggingface_dataset(self) -> Optional["DatasetDict"]:
        """Convert to HuggingFace DatasetDict."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace datasets not available. Install with: pip install datasets")
            return None
        
        # Define features schema
        features = Features({
            "expression": Value("string"),
            "evaluation": Value("string"),
            "tokens": Sequence(Value("string")),
            "metadata": {
                "token_count": Value("int32"),
                "operator_count": Value("int32"),
                "literal_count": Value("int32"),
                "nesting_depth": Value("int32"),
                "has_negation": Value("bool"),
                "is_error": Value("bool"),
                "complexity_score": Value("float32")
            },
            "reasoning_steps": Sequence({
                "step": Value("int32"),
                "operation": Value("string"),
                "description": Value("string")
            }),
            "error_type": Value("string"),
            "created_at": Value("string")
        })
        
        dataset_splits = {}
        for split_name, examples in self.splits.items():
            if not examples:
                continue
                
            split_data = [ex.to_dict() for ex in examples]
            dataset_splits[split_name] = Dataset.from_list(split_data, features=features)
        
        return DatasetDict(dataset_splits)
    
    def create_dataset_card(self, output_path: Union[str, Path]):
        """Create a HuggingFace dataset card."""
        output_path = Path(output_path)
        
        card_content = f"""---
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
    num_bytes: {self._estimate_bytes("train")}
    num_examples: {len(self.splits["train"])}
  - name: validation
    num_bytes: {self._estimate_bytes("validation")}
    num_examples: {len(self.splits["validation"])}
  - name: test
    num_bytes: {self._estimate_bytes("test")}
    num_examples: {len(self.splits["test"])}
license: {self.license}
task_categories:
- text-classification
- logical-reasoning
language:
- en
size_categories:
- {self._get_size_category()}
---

# {self.name}

{self.description}

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

{self._format_split_info()}

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{self.name}")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"] 
test_data = dataset["test"]

# Example usage
for example in train_data:
    print(f"Expression: {{example['expression']}}")
    print(f"Result: {{example['evaluation']}}")
    print(f"Complexity: {{example['metadata']['complexity_score']}}")
```

## Dataset Creation

This dataset was generated using the BoLoCo toolkit with the following configuration:

```json
{json.dumps(self.metadata["generation_config"], indent=2)}
```

## Statistics

{self._format_statistics()}

## Citation

```bibtex
@misc{{boloco-enhanced,
  title={{BoLoCo Enhanced: Boolean Logic Expression Dataset}},
  author={{{", ".join(self.authors)}}},
  year={{2024}},
  version={{{self.version}}},
  url={{https://github.com/klusai/boloco}}
}}
```

## License

{self.license}
"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        
        logger.info(f"Created dataset card at {output_path}")
    
    def _estimate_bytes(self, split: str) -> int:
        """Estimate dataset size in bytes for the given split."""
        if not self.splits[split]:
            return 0
        
        # Rough estimate based on JSON serialization
        sample_size = len(json.dumps(self.splits[split][0].to_dict()))
        return sample_size * len(self.splits[split])
    
    def _get_size_category(self) -> str:
        """Get HuggingFace size category."""
        total = sum(len(examples) for examples in self.splits.values())
        
        if total < 1000:
            return "n<1K"
        elif total < 10000:
            return "1K<n<10K"
        elif total < 100000:
            return "10K<n<100K"
        elif total < 1000000:
            return "100K<n<1M"
        else:
            return "n>1M"
    
    def _format_split_info(self) -> str:
        """Format split information for dataset card."""
        info = []
        for split_name, examples in self.splits.items():
            if examples:
                info.append(f"- **{split_name}**: {len(examples)} examples")
        return "\n".join(info)
    
    def _format_statistics(self) -> str:
        """Format statistics for dataset card."""
        if "statistics" not in self.metadata:
            return "Statistics not available."
        
        stats_text = []
        for split_name, stats in self.metadata["statistics"].items():
            stats_text.append(f"### {split_name.title()} Split")
            stats_text.append(f"- Total examples: {stats['total_examples']}")
            stats_text.append(f"- True evaluations: {stats['true_evaluations']}")
            stats_text.append(f"- False evaluations: {stats['false_evaluations']}")
            stats_text.append(f"- Error evaluations: {stats['error_evaluations']}")
            stats_text.append(f"- Average token count: {stats['avg_token_count']:.2f}")
            stats_text.append(f"- Average complexity: {stats['avg_complexity']:.2f}")
            stats_text.append("")
        
        return "\n".join(stats_text)


def convert_legacy_to_enhanced(
    legacy_file: Union[str, Path],
    output_path: Union[str, Path],
    format: str = "jsonl"
) -> BoLoCoDataset:
    """
    Convert legacy BoLoCo format to enhanced format.
    
    Args:
        legacy_file: Path to legacy format file
        output_path: Output path for enhanced format
        format: Output format ("jsonl", "json", or "hf")
    
    Returns:
        BoLoCoDataset instance
    """
    legacy_file = Path(legacy_file)
    dataset = BoLoCoDataset(name=f"boloco-converted-{legacy_file.stem}")
    
    with open(legacy_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                example = BoLoCoExample.from_legacy_format(line)
                dataset.add_example(example, "train")  # Default to train split
            except ValueError as e:
                logger.warning(f"Skipping invalid line {line_num}: {e}")
    
    # Save in requested format
    if format == "jsonl":
        dataset.save_jsonl(output_path)
    elif format == "json":
        dataset.save_json(output_path)
    elif format == "hf" and HF_AVAILABLE:
        hf_dataset = dataset.to_huggingface_dataset()
        hf_dataset.save_to_disk(output_path)
    
    logger.info(f"Converted {len(dataset.splits['train'])} examples from {legacy_file}")
    return dataset