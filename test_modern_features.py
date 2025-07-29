#!/usr/bin/env python3
"""
Test script for BoLoCo Enhanced modern features.

This script demonstrates the new modern functionality including:
- Modern data formats with rich metadata
- HuggingFace integration
- Format conversion capabilities
- Validation features
"""

import tempfile
import json
from pathlib import Path
import sys
import os

# Add the boloco package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

def test_modern_example_creation():
    """Test creating a modern BoLoCo example."""
    print("🧪 Testing ModernBoLoCoExample creation...")
    
    from boloco.modern_formats import ModernBoLoCoExample
    
    # Create a simple example
    example = ModernBoLoCoExample(
        expression="T OR F",
        evaluation="T",
        tokens=["T", "OR", "F"]
    )
    
    # Check metadata auto-computation
    assert example.metadata["token_count"] == 3
    assert example.metadata["operator_count"] == 1
    assert example.metadata["literal_count"] == 2
    assert example.metadata["nesting_depth"] == 0
    assert not example.metadata["has_negation"]
    assert not example.metadata["is_error"]
    
    # Test JSON serialization
    example_dict = example.to_dict()
    assert "created_at" in example_dict
    assert example_dict["expression"] == "T OR F"
    
    # Test legacy format conversion
    legacy_format = example.to_legacy_format()
    assert legacy_format == "<s> T OR F <eval/> T </s>"
    
    print("✅ ModernBoLoCoExample tests passed!")


def test_legacy_conversion():
    """Test converting legacy format to modern."""
    print("🧪 Testing legacy format conversion...")
    
    from boloco.modern_formats import ModernBoLoCoExample
    
    # Test legacy format parsing
    legacy_line = "<s> ( T AND F ) <eval/> F </s>"
    example = ModernBoLoCoExample.from_legacy_format(legacy_line)
    
    assert example.expression == "( T AND F )"
    assert example.evaluation == "F"
    assert example.tokens == ["(", "T", "AND", "F", ")"]
    assert example.metadata["token_count"] == 5
    assert example.metadata["operator_count"] == 1
    assert example.metadata["nesting_depth"] == 1
    
    print("✅ Legacy conversion tests passed!")


def test_modern_dataset():
    """Test modern dataset functionality."""
    print("🧪 Testing ModernBoLoCoDataset...")
    
    from boloco.modern_formats import ModernBoLoCoDataset, ModernBoLoCoExample
    
    # Create dataset
    dataset = ModernBoLoCoDataset(
        name="test-dataset",
        version="1.0.0",
        description="Test dataset for validation"
    )
    
    # Add some examples
    examples = [
        ModernBoLoCoExample("T", "T", ["T"]),
        ModernBoLoCoExample("F", "F", ["F"]),
        ModernBoLoCoExample("T OR F", "T", ["T", "OR", "F"]),
        ModernBoLoCoExample("INVALID", "ERR", ["INVALID"], error_type="syntax_error")
    ]
    
    for i, example in enumerate(examples):
        split = ["train", "validation", "test"][i % 3]
        dataset.add_example(example, split)
    
    # Update metadata
    config = {"max_tokens": 5, "error_ratio": 0.25}
    dataset.update_metadata(config)
    
    # Check statistics
    stats = dataset.metadata["statistics"]
    assert "train" in stats
    assert stats["train"]["total_examples"] >= 1
    
    print("✅ ModernBoLoCoDataset tests passed!")


def test_file_operations():
    """Test file I/O operations."""
    print("🧪 Testing file operations...")
    
    from boloco.modern_formats import ModernBoLoCoDataset, ModernBoLoCoExample
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create dataset
        dataset = ModernBoLoCoDataset(name="test-file-ops")
        example = ModernBoLoCoExample("T OR F", "T", ["T", "OR", "F"])
        dataset.add_example(example, "train")
        dataset.update_metadata({"test": True})
        
        # Test JSON export
        json_path = temp_path / "test.json"
        dataset.save_json(json_path)
        assert json_path.exists()
        
        # Validate JSON content
        with open(json_path, "r") as f:
            data = json.load(f)
        assert data["name"] == "test-file-ops"
        assert "train" in data["splits"]
        
        # Test JSONL export
        jsonl_path = temp_path / "test.jsonl"
        dataset.save_jsonl(jsonl_path, split="train")
        assert (temp_path / "test_train.jsonl").exists()
        
        # Test legacy format export
        legacy_dir = temp_path / "legacy"
        dataset.save_legacy_format(legacy_dir)
        assert (legacy_dir / "boloco-train-legacy.txt").exists()
        
        # Test dataset card creation
        card_path = temp_path / "README.md"
        dataset.create_dataset_card(card_path)
        assert card_path.exists()
        
        # Validate card content
        with open(card_path, "r") as f:
            card_content = f.read()
        assert "test-file-ops" in card_content
        assert "## Dataset Description" in card_content
    
    print("✅ File operations tests passed!")


def test_huggingface_integration():
    """Test HuggingFace integration (if available)."""
    print("🧪 Testing HuggingFace integration...")
    
    try:
        from datasets import Dataset, DatasetDict
        HF_AVAILABLE = True
    except ImportError:
        print("⚠️  HuggingFace datasets not available, skipping integration test")
        return
    
    from boloco.modern_formats import ModernBoLoCoDataset, ModernBoLoCoExample
    
    # Create dataset
    dataset = ModernBoLoCoDataset(name="test-hf")
    examples = [
        ModernBoLoCoExample("T", "T", ["T"]),
        ModernBoLoCoExample("F", "F", ["F"]),
        ModernBoLoCoExample("T AND F", "F", ["T", "AND", "F"])
    ]
    
    for example in examples:
        dataset.add_example(example, "train")
    
    # Convert to HuggingFace format
    hf_dataset = dataset.to_huggingface_dataset()
    
    if hf_dataset:
        assert isinstance(hf_dataset, DatasetDict)
        assert "train" in hf_dataset
        assert len(hf_dataset["train"]) == 3
        
        # Check schema
        features = hf_dataset["train"].features
        assert "expression" in features
        assert "evaluation" in features
        assert "metadata" in features
        
        print("✅ HuggingFace integration tests passed!")
    else:
        print("⚠️  HuggingFace integration returned None")


def test_modern_cli_config_validation():
    """Test CLI configuration validation."""
    print("🧪 Testing CLI configuration validation...")
    
    from boloco.modern_cli import validate_config
    from argparse import Namespace
    
    # Create valid args
    args = Namespace(
        max_tokens=5,
        error_ratio=0.1,
        train_ratio=0.7,
        validate_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # Test valid configuration
    config = validate_config(args)
    assert config["max_tokens"] == 5
    assert config["error_ratio"] == 0.1
    assert config["seed"] == 42
    
    # Test auto-adjustment of test ratio
    args.test_ratio = 0.2  # This will make total > 1
    config = validate_config(args)
    assert abs(config["train_ratio"] + config["validate_ratio"] + config["test_ratio"] - 1.0) < 1e-10
    
    # Test validation errors
    args.max_tokens = -1
    try:
        validate_config(args)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    print("✅ CLI configuration validation tests passed!")


def run_all_tests():
    """Run all tests."""
    print("🚀 Running BoLoCo Enhanced test suite...\n")
    
    tests = [
        test_modern_example_creation,
        test_legacy_conversion,
        test_modern_dataset,
        test_file_operations,
        test_huggingface_integration,
        test_modern_cli_config_validation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
            print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! BoLoCo Enhanced is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return failed == 0


def demo_modern_features():
    """Demonstrate modern features with example usage."""
    print("🎯 BoLoCo Enhanced Feature Demo\n")
    
    from boloco.modern_formats import ModernBoLoCoDataset, ModernBoLoCoExample
    
    # Create a demo dataset
    print("📝 Creating demo dataset...")
    dataset = ModernBoLoCoDataset(
        name="boloco-demo",
        version="1.0.0",
        description="Demonstration of BoLoCo Enhanced features"
    )
    
    # Add various examples
    demo_expressions = [
        ("T", "T", ["T"]),
        ("F", "F", ["F"]),
        ("T OR F", "T", ["T", "OR", "F"]),
        ("T AND F", "F", ["T", "AND", "F"]),
        ("NOT T", "F", ["NOT", "T"]),
        ("( T OR F ) AND NOT F", "T", ["(", "T", "OR", "F", ")", "AND", "NOT", "F"]),
        ("INVALID_EXPR", "ERR", ["INVALID_EXPR"])
    ]
    
    for i, (expr, eval_result, tokens) in enumerate(demo_expressions):
        example = ModernBoLoCoExample(
            expression=expr,
            evaluation=eval_result, 
            tokens=tokens,
            error_type="syntax_error" if eval_result == "ERR" else None
        )
        
        # Distribute across splits
        split = ["train", "validation", "test"][i % 3]
        dataset.add_example(example, split)
    
    # Update metadata
    dataset.update_metadata({
        "demo": True,
        "max_tokens": 8,
        "error_ratio": 1/7
    })
    
    # Show statistics
    print("\n📊 Dataset Statistics:")
    stats = dataset.metadata["statistics"]
    for split_name, split_stats in stats.items():
        if split_stats["total_examples"] > 0:
            print(f"  {split_name.title()}: {split_stats['total_examples']} examples, "
                  f"avg complexity: {split_stats['avg_complexity']:.2f}")
    
    # Demonstrate format exports
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"\n💾 Exporting to various formats in {temp_path}...")
        
        # JSON export
        dataset.save_json(temp_path / "demo.json")
        print("  ✓ JSON format exported")
        
        # JSONL export
        dataset.save_jsonl(temp_path / "demo.jsonl")
        print("  ✓ JSONL format exported")
        
        # Legacy format
        dataset.save_legacy_format(temp_path / "legacy")
        print("  ✓ Legacy format exported")
        
        # Dataset card
        dataset.create_dataset_card(temp_path / "README.md")
        print("  ✓ Dataset card created")
        
        # Show file sizes
        print("\n📁 Generated files:")
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  {file_path.name}: {size} bytes")
    
    print("\n🎉 Demo completed! BoLoCo Enhanced is working correctly.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BoLoCo Enhanced features")
    parser.add_argument("--demo", action="store_true", help="Run feature demonstration")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    
    args = parser.parse_args()
    
    if not args.demo and not args.test:
        # Run both by default
        success = run_all_tests()
        print("\n" + "="*50 + "\n")
        demo_modern_features()
    elif args.test:
        success = run_all_tests()
    elif args.demo:
        demo_modern_features()