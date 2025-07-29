"""
Enhanced CLI interface for BoLoCo with improved UX and format support.

This module provides an enhanced command-line interface with:
- Rich output formatting
- Progress bars and status indicators  
- Input validation and error handling
- Support for JSON/JSONL and HuggingFace formats
- HuggingFace integration
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Optional imports for rich CLI experience
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = Progress = Table = Panel = Text = None
    rprint = print

from .enhanced import BoLoCoDataset, BoLoCoExample

logger = logging.getLogger("boloco.cli")


def generate_logic_expressions(max_tokens):
    """Generate valid Boolean logic expressions."""
    import random
    
    # Basic tokens
    literals = ["T", "F"]
    operators = ["AND", "OR", "NOT"]
    
    def generate_single_expression(target_tokens):
        """Generate a single expression with approximately target_tokens tokens."""
        if target_tokens == 1:
            return random.choice(literals)
        
        if target_tokens == 2:
            if random.random() < 0.5:
                return f"NOT {random.choice(literals)}"
            else:
                return random.choice(literals)
        
        # For longer expressions, build recursively
        if random.random() < 0.3:  # Use parentheses sometimes
            left_tokens = max(1, target_tokens // 2 - 1)
            right_tokens = target_tokens - left_tokens - 3  # -3 for operator and parentheses
            if right_tokens < 1:
                right_tokens = 1
                left_tokens = target_tokens - right_tokens - 3
            
            if left_tokens >= 1 and right_tokens >= 1:
                left = generate_single_expression(left_tokens)
                right = generate_single_expression(right_tokens)
                op = random.choice(["AND", "OR"])
                return f"( {left} {op} {right} )"
        
        # Binary operation without parentheses
        left_tokens = max(1, target_tokens // 2)
        right_tokens = target_tokens - left_tokens - 1  # -1 for operator
        if right_tokens < 1:
            right_tokens = 1
            left_tokens = target_tokens - right_tokens - 1
        
        if left_tokens >= 1 and right_tokens >= 1:
            left = generate_single_expression(left_tokens)
            right = generate_single_expression(right_tokens)
            op = random.choice(["AND", "OR"])
            return f"{left} {op} {right}"
        
        return random.choice(literals)
    
    expressions = set()
    attempts = 0
    max_attempts = 1000
    
    while len(expressions) < 100 and attempts < max_attempts:
        attempts += 1
        target_tokens = random.randint(1, max_tokens)
        expr = generate_single_expression(target_tokens)
        
        # Validate token count
        actual_tokens = len(expr.split())
        if actual_tokens <= max_tokens:
            expressions.add(expr)
    
    return list(expressions)


def generate_error_expressions(valid_expressions, num_errors):
    """Generate erroneous Boolean logic expressions."""
    import random
    
    if num_errors == 0:
        return []
    
    errors = []
    operators = ["AND", "OR", "NOT"]
    literals = ["T", "F"]
    
    for _ in range(num_errors):
        if random.random() < 0.5 and valid_expressions:
            # Corrupt a valid expression
            expr = random.choice(valid_expressions)
            tokens = expr.split()
            
            if len(tokens) > 1:
                # Random corruption strategies
                corruption = random.choice([
                    "duplicate_operator",
                    "missing_operand", 
                    "invalid_token",
                    "unbalanced_parens"
                ])
                
                if corruption == "duplicate_operator":
                    # Add extra operator
                    pos = random.randint(0, len(tokens))
                    tokens.insert(pos, random.choice(operators))
                elif corruption == "missing_operand":
                    # Remove a literal
                    literal_indices = [i for i, t in enumerate(tokens) if t in literals]
                    if literal_indices:
                        tokens.pop(random.choice(literal_indices))
                elif corruption == "invalid_token":
                    # Replace with invalid token
                    pos = random.randint(0, len(tokens) - 1)
                    tokens[pos] = "INVALID"
                elif corruption == "unbalanced_parens":
                    # Add unmatched parenthesis
                    tokens.append("(")
                
                errors.append(" ".join(tokens))
            else:
                errors.append("INVALID")
        else:
            # Generate completely invalid expression
            invalid_patterns = [
                "AND OR",
                "NOT",
                "T AND",
                "OR F",
                "( T",
                "F )",
                "INVALID TOKEN",
                "T AND AND F"
            ]
            errors.append(random.choice(invalid_patterns))
    
    return errors


def eval_expression(expr):
    """Evaluate a Boolean logic expression."""
    try:
        # Replace tokens with Python equivalents
        expr = expr.replace("T", "True").replace("F", "False")
        expr = expr.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
        
        # Basic validation - check for valid tokens only
        import re
        valid_pattern = r'^[\s\(\)TrueFalseandondt]+$'
        if not re.match(valid_pattern, expr):
            return None
            
        # Evaluate safely
        result = eval(expr)
        return result if isinstance(result, bool) else None
    except:
        return None


def split_dataset(data, train_ratio=0.7, validate_ratio=0.15, test_ratio=0.15):
    """Split dataset into train, validation, and test sets."""
    import random
    
    data = list(data)
    random.shuffle(data)
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * validate_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data


class BoLoCoGenerator:
    """Enhanced dataset generator with rich features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.console = Console() if RICH_AVAILABLE else None
        
    def generate_dataset(self) -> BoLoCoDataset:
        """Generate an enhanced BoLoCo dataset."""
        if RICH_AVAILABLE:
            self.console.print(Panel(
                "[bold green]BoLoCo Enhanced Dataset Generation[/bold green]",
                subtitle="Generating Boolean logic expressions with rich metadata"
            ))
        else:
            print("=== BoLoCo Enhanced Dataset Generation ===")
        
        # Create dataset
        dataset = BoLoCoDataset(
            name=self.config.get("name", "boloco-enhanced"),
            version=self.config.get("version", "2.0.0"),
            description=self.config.get("description", "Boolean Logic Expression Dataset with Enhanced Metadata"),
            authors=self.config.get("authors", ["BoLoCo Contributors"]),
            license=self.config.get("license", "MIT")
        )
        
        # Update metadata with generation config
        dataset.update_metadata(self.config)
        
        # Generate expressions
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Generating expressions...", total=100)
                
                # Generate valid expressions
                progress.update(task, description="Generating valid expressions...", advance=20)
                valid_expressions = list(generate_logic_expressions(
                    self.config["max_tokens"]
                ))
                
                # Generate errors
                progress.update(task, description="Generating error expressions...", advance=20)
                total_errors = int(len(valid_expressions) * self.config["error_ratio"])
                errors = generate_error_expressions(valid_expressions, total_errors)
                
                # Create examples
                progress.update(task, description="Creating enhanced examples...", advance=20)
                enhanced_examples = []
                
                # Convert valid expressions
                for expr in valid_expressions:
                    tokens = expr.split()
                    example = BoLoCoExample(
                        expression=expr,
                        evaluation="T" if self._evaluate_expression(expr) else "F",
                        tokens=tokens
                    )
                    enhanced_examples.append(example)
                
                # Convert error expressions
                for expr in errors:
                    tokens = expr.split() if isinstance(expr, str) else list(expr)
                    example = BoLoCoExample(
                        expression=" ".join(tokens) if isinstance(expr, (list, tuple)) else expr,
                        evaluation="ERR",
                        tokens=tokens,
                        error_type="syntax_error"  # Could be enhanced with more specific error types
                    )
                    enhanced_examples.append(example)
                
                # Split dataset
                progress.update(task, description="Splitting dataset...", advance=20)
                train_examples, val_examples, test_examples = self._split_examples(
                    enhanced_examples,
                    self.config["train_ratio"],
                    self.config["validate_ratio"]
                )
                
                # Add to dataset
                progress.update(task, description="Finalizing dataset...", advance=20)
                dataset.add_examples(train_examples, "train")
                dataset.add_examples(val_examples, "validation")  
                dataset.add_examples(test_examples, "test")
                
                progress.update(task, description="Dataset generation complete!", completed=100)
        else:
            print("Generating valid expressions...")
            valid_expressions = list(generate_logic_expressions(
                self.config["max_tokens"]
            ))
            
            print(f"Generated {len(valid_expressions)} valid expressions")
            
            print("Generating error expressions...")
            total_errors = int(len(valid_expressions) * self.config["error_ratio"])
            errors = generate_error_expressions(valid_expressions, total_errors)
            
            print("Creating enhanced examples...")
            enhanced_examples = []
            
            # Convert expressions (simplified without progress)
            for expr in valid_expressions:
                tokens = expr.split()
                example = BoLoCoExample(
                    expression=expr,
                    evaluation="T" if self._evaluate_expression(expr) else "F",
                    tokens=tokens
                )
                enhanced_examples.append(example)
            
            for expr in errors:
                tokens = expr.split() if isinstance(expr, str) else list(expr)
                example = BoLoCoExample(
                    expression=" ".join(tokens) if isinstance(expr, (list, tuple)) else expr,
                    evaluation="ERR",
                    tokens=tokens,
                    error_type="syntax_error"
                )
                enhanced_examples.append(example)
            
            print("Splitting dataset...")
            train_examples, val_examples, test_examples = self._split_examples(
                enhanced_examples,
                self.config["train_ratio"],
                self.config["validate_ratio"]
            )
            
            dataset.add_examples(train_examples, "train")
            dataset.add_examples(val_examples, "validation")
            dataset.add_examples(test_examples, "test")
            
            print("Dataset generation complete!")
        
        return dataset
    
    def _evaluate_expression(self, expr: str) -> bool:
        """Evaluate a Boolean expression."""
        result = eval_expression(expr)
        return result if isinstance(result, bool) else False
    
    def _split_examples(self, examples: List[BoLoCoExample], train_ratio: float, val_ratio: float):
        """Split examples into train/val/test sets."""
        import random
        
        # Separate valid and error examples
        valid_examples = [ex for ex in examples if ex.evaluation != "ERR"]
        error_examples = [ex for ex in examples if ex.evaluation == "ERR"]
        
        # Split each type proportionally
        def split_list(items, train_r, val_r):
            random.shuffle(items)
            n = len(items)
            train_end = int(n * train_r)
            val_end = train_end + int(n * val_r)
            return items[:train_end], items[train_end:val_end], items[val_end:]
        
        train_valid, val_valid, test_valid = split_list(valid_examples, train_ratio, val_ratio)
        train_error, val_error, test_error = split_list(error_examples, train_ratio, val_ratio)
        
        return (
            train_valid + train_error,
            val_valid + val_error,
            test_valid + test_error
        )


def print_dataset_stats(dataset: BoLoCoDataset):
    """Print dataset statistics in a formatted way."""
    if RICH_AVAILABLE:
        console = Console()
        
        # Main stats table
        table = Table(title="Dataset Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Split", style="cyan", no_wrap=True)
        table.add_column("Examples", justify="right", style="green")
        table.add_column("True", justify="right", style="blue")
        table.add_column("False", justify="right", style="yellow")
        table.add_column("Errors", justify="right", style="red")
        table.add_column("Avg Complexity", justify="right", style="purple")
        
        stats = dataset.metadata.get("statistics", {})
        for split_name, split_stats in stats.items():
            table.add_row(
                split_name.title(),
                str(split_stats["total_examples"]),
                str(split_stats["true_evaluations"]),
                str(split_stats["false_evaluations"]),
                str(split_stats["error_evaluations"]),
                f"{split_stats['avg_complexity']:.2f}"
            )
        
        console.print(table)
        
        # Operator distribution
        console.print("\n[bold]Operator Distribution:[/bold]")
        for split_name, split_stats in stats.items():
            if "operator_distribution" in split_stats:
                op_dist = split_stats["operator_distribution"]
                console.print(f"  {split_name.title()}: AND={op_dist.get('AND', 0)}, OR={op_dist.get('OR', 0)}, NOT={op_dist.get('NOT', 0)}")
    else:
        print("\n=== Dataset Statistics ===")
        stats = dataset.metadata.get("statistics", {})
        for split_name, split_stats in stats.items():
            print(f"\n{split_name.title()} Split:")
            print(f"  Total examples: {split_stats['total_examples']}")
            print(f"  True evaluations: {split_stats['true_evaluations']}")
            print(f"  False evaluations: {split_stats['false_evaluations']}")
            print(f"  Error evaluations: {split_stats['error_evaluations']}")
            print(f"  Average complexity: {split_stats['avg_complexity']:.2f}")
            
            if "operator_distribution" in split_stats:
                op_dist = split_stats["operator_distribution"]
                print(f"  Operators: AND={op_dist.get('AND', 0)}, OR={op_dist.get('OR', 0)}, NOT={op_dist.get('NOT', 0)}")


def validate_config(args) -> Dict[str, Any]:
    """Validate and normalize configuration from CLI arguments."""
    config = {}
    
    # Basic validation
    if args.max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    
    if not (0 <= args.error_ratio <= 1):
        raise ValueError("error_ratio must be between 0 and 1")
    
    if not (0 < args.train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")
    
    if not (0 < args.validate_ratio < 1):
        raise ValueError("validate_ratio must be between 0 and 1")
    
    if args.train_ratio + args.validate_ratio + args.test_ratio != 1.0:
        # Auto-adjust test ratio
        args.test_ratio = 1.0 - args.train_ratio - args.validate_ratio
        if args.test_ratio <= 0:
            raise ValueError("train_ratio + validate_ratio must be less than 1")
    
    # Normalize config
    config.update({
        "max_tokens": args.max_tokens,
        "error_ratio": args.error_ratio,
        "train_ratio": args.train_ratio,
        "validate_ratio": args.validate_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "name": getattr(args, "name", "boloco-enhanced"),
        "version": getattr(args, "version", "2.0.0"),
        "description": getattr(args, "description", "Boolean Logic Expression Dataset with Enhanced Metadata"),
        "authors": getattr(args, "authors", ["BoLoCo Contributors"]),
        "license": getattr(args, "license", "MIT")
    })
    
    return config


def cmd_generate(args):
    """Generate a new dataset in enhanced format."""
    try:
        config = validate_config(args)
        generator = BoLoCoGenerator(config)
        dataset = generator.generate_dataset()
        
        # Save in requested formats
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.format in ["json", "all"]:
            dataset.save_json(output_dir / "dataset.json")
        
        if args.format in ["jsonl", "all"]:
            dataset.save_jsonl(output_dir / "dataset.jsonl")
        

        
        if args.format in ["hf", "all"]:
            hf_dataset = dataset.to_huggingface_dataset()
            if hf_dataset:
                hf_dataset.save_to_disk(output_dir / "hf_dataset")
            
        # Create dataset card
        dataset.create_dataset_card(output_dir / "README.md")
        
        # Print statistics
        print_dataset_stats(dataset)
        
        if RICH_AVAILABLE:
            rprint(f"\n[bold green]✓[/bold green] Dataset saved to {output_dir}")
        else:
            print(f"\n✓ Dataset saved to {output_dir}")
            
    except Exception as e:
        if RICH_AVAILABLE:
            rprint(f"[bold red]Error:[/bold red] {e}")
        else:
            print(f"Error: {e}")
        sys.exit(1)





def create_parser():
    """Create the argument parser for the modern CLI."""
    parser = argparse.ArgumentParser(
        description="BoLoCo Enhanced: Boolean Logic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a new dataset in JSON format
  python -m boloco.cli generate --max-tokens 7 --output-dir ./data --format json

  # Generate with specific parameters
  python -m boloco.cli generate --max-tokens 10 --error-ratio 0.1 --format all
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a new dataset")
    gen_parser.add_argument("--max-tokens", type=int, default=5,
                           help="Maximum number of tokens per expression")
    gen_parser.add_argument("--error-ratio", type=float, default=0.05,
                           help="Ratio of error examples to include")
    gen_parser.add_argument("--train-ratio", type=float, default=0.7,
                           help="Ratio for training split")
    gen_parser.add_argument("--validate-ratio", type=float, default=0.15,
                           help="Ratio for validation split")
    gen_parser.add_argument("--test-ratio", type=float, default=0.15,
                           help="Ratio for test split")
    gen_parser.add_argument("--seed", type=int, default=42,
                           help="Random seed for reproducibility")
    gen_parser.add_argument("--output-dir", type=str, default="./data",
                           help="Output directory for generated dataset")
    gen_parser.add_argument("--format", choices=["json", "jsonl", "hf", "all"],
                           default="all", help="Output format(s)")
    gen_parser.add_argument("--name", type=str, default="boloco-enhanced",
                           help="Dataset name")
    gen_parser.add_argument("--version", type=str, default="2.0.0",
                           help="Dataset version")
    gen_parser.set_defaults(func=cmd_generate)
    

    
    return parser


def main():
    """Main entry point for the enhanced CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Call the appropriate command function
    args.func(args)


if __name__ == "__main__":
    main()