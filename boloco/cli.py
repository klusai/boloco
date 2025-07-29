"""
Enhanced CLI interface for BoLoCo with improved UX and format support.

This module provides an enhanced command-line interface with:
- Rich output formatting
- Progress bars and status indicators  
- Input validation and error handling
- Support for both legacy and enhanced formats
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

from .enhanced import BoLoCoDataset, BoLoCoExample, convert_legacy_to_enhanced
from .boloco import generate_logic_expressions, split_dataset, generate_error_expressions

logger = logging.getLogger("boloco.cli")


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
                    self.config["max_tokens"],
                    self.config.get("max_not_depth", 2),
                    self.config.get("max_paren_depth", 2)
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
                self.config["max_tokens"],
                self.config.get("max_not_depth", 2),
                self.config.get("max_paren_depth", 2)
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
        # Import eval_expression from boloco module
        from .boloco import eval_expression
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
        
        if args.format in ["legacy", "all"]:
            dataset.save_legacy_format(output_dir / "legacy")
        
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


def cmd_convert(args):
    """Convert legacy format to enhanced format."""
    try:
        legacy_file = Path(args.input_file)
        if not legacy_file.exists():
            raise FileNotFoundError(f"Input file not found: {legacy_file}")
        
        output_path = Path(args.output_path)
        
        if RICH_AVAILABLE:
            rprint(f"[cyan]Converting[/cyan] {legacy_file} → {output_path}")
        else:
            print(f"Converting {legacy_file} → {output_path}")
        
        dataset = convert_legacy_to_enhanced(legacy_file, output_path, args.format)
        
        if args.create_card:
            card_path = output_path.parent / "README.md"
            dataset.create_dataset_card(card_path)
            
        if RICH_AVAILABLE:
            rprint(f"[bold green]✓[/bold green] Conversion complete!")
        else:
            print("✓ Conversion complete!")
            
    except Exception as e:
        if RICH_AVAILABLE:
            rprint(f"[bold red]Error:[/bold red] {e}")
        else:
            print(f"Error: {e}")
        sys.exit(1)


def cmd_validate(args):
    """Validate a dataset file."""
    try:
        file_path = Path(args.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Basic validation
            required_fields = ["name", "version", "splits"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            if RICH_AVAILABLE:
                rprint(f"[bold green]✓[/bold green] Valid BoLoCo Enhanced dataset")
                rprint(f"  Name: {data['name']}")
                rprint(f"  Version: {data['version']}")
                rprint(f"  Splits: {list(data['splits'].keys())}")
            else:
                print("✓ Valid BoLoCo Enhanced dataset")
                print(f"  Name: {data['name']}")
                print(f"  Version: {data['version']}")
                print(f"  Splits: {list(data['splits'].keys())}")
                
        elif file_path.suffix == ".txt":
            # Validate legacy format
            valid_lines = 0
            invalid_lines = 0
            
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        BoLoCoExample.from_legacy_format(line)
                        valid_lines += 1
                    except ValueError:
                        invalid_lines += 1
                        if args.verbose:
                            print(f"Invalid line {line_num}: {line}")
            
            if RICH_AVAILABLE:
                if invalid_lines == 0:
                    rprint(f"[bold green]✓[/bold green] Valid legacy BoLoCo dataset")
                else:
                    rprint(f"[yellow]⚠[/yellow] Legacy dataset with issues")
                rprint(f"  Valid lines: {valid_lines}")
                rprint(f"  Invalid lines: {invalid_lines}")
            else:
                status = "✓ Valid" if invalid_lines == 0 else "⚠ Has issues"
                print(f"{status} legacy BoLoCo dataset")
                print(f"  Valid lines: {valid_lines}")
                print(f"  Invalid lines: {invalid_lines}")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
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
  # Generate a new dataset
  python -m boloco.modern_cli generate --max-tokens 7 --output-dir ./data

  # Convert legacy format
  python -m boloco.modern_cli convert legacy_file.txt modern_file.jsonl

  # Validate a dataset
  python -m boloco.modern_cli validate dataset.json
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
    gen_parser.add_argument("--format", choices=["json", "jsonl", "legacy", "hf", "all"],
                           default="all", help="Output format(s)")
    gen_parser.add_argument("--name", type=str, default="boloco-enhanced",
                           help="Dataset name")
    gen_parser.add_argument("--version", type=str, default="2.0.0",
                           help="Dataset version")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Convert command  
    conv_parser = subparsers.add_parser("convert", help="Convert legacy format to enhanced")
    conv_parser.add_argument("input_file", help="Input legacy format file")
    conv_parser.add_argument("output_path", help="Output path for enhanced format")
    conv_parser.add_argument("--format", choices=["json", "jsonl", "hf"],
                            default="jsonl", help="Output format")
    conv_parser.add_argument("--create-card", action="store_true",
                            help="Create dataset card")
    conv_parser.set_defaults(func=cmd_convert)
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a dataset file")
    val_parser.add_argument("file_path", help="Path to dataset file")
    val_parser.add_argument("--verbose", action="store_true",
                           help="Show detailed validation errors")
    val_parser.set_defaults(func=cmd_validate)
    
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