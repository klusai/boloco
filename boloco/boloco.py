import argparse
import glob
import logging
import os
import random
import time
import numpy as np
import warnings

# Defines tokens used in the dataset
SOS_TOKEN = "<s>"  # Start of sequence token
EVAL_TOKEN = "<eval/>"  # Evaluation token separating expression and result
EOS_TOKEN = "</s>"  # End of sequence token
ERROR_TOKEN = "<err/>"  # Error token for invalid expressions
INVALID_TOKEN = "<inv/>"  # Invalid token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("boloco")


def generate_logic_expressions(n, max_not_depth=2, max_paren_depth=2):
    """
    Generate Boolean logic expressions with constraints on nesting and complexity.

    Args:
        n (int): Maximum number of tokens in expressions.
        max_not_depth (int): Maximum allowed depth of consecutive `NOT` operators.
        max_paren_depth (int): Maximum nesting depth of parentheses.

    Returns:
        set: A set of valid Boolean logic expressions.
    """
    start_time = time.time()
    logger.info(f"Generating logic expressions with up to {n} tokens.")

    # Define tokens
    LITERALS = ["T", "F"]
    BINARY_OPERATORS = ["AND", "OR"]
    UNARY_OPERATORS = ["NOT"]
    OPEN_PAREN = "("
    CLOSE_PAREN = ")"

    valid_expressions = set()

    def build(
        current,
        tokens_left,
        expect_literal,
        open_parens,
        current_not_depth,
        max_paren_depth,
    ):
        if tokens_left == 0:
            # Completed an expression if no open parentheses and no pending literal
            if open_parens == 0 and not expect_literal:
                valid_expressions.add(" ".join(current))
            return

        # Add literals if expecting a literal
        if expect_literal:
            for literal in LITERALS:
                build(
                    current + [literal],
                    tokens_left - 1,
                    False,
                    open_parens,
                    0,  # Reset NOT depth after a literal
                    max_paren_depth,
                )

        # Add unary operator (NOT) if expecting a literal and within depth limit
        if expect_literal and current_not_depth < max_not_depth:
            for unary_operator in UNARY_OPERATORS:
                build(
                    current + [unary_operator],
                    tokens_left - 1,
                    True,
                    open_parens,
                    current_not_depth + 1,
                    max_paren_depth,
                )

        # Add binary operators (AND, OR) if expecting an operator
        # Only add if the last token is not an operator or open parenthesis
        if (
            not expect_literal
            and current
            and current[-1] not in ([OPEN_PAREN] + BINARY_OPERATORS + UNARY_OPERATORS)
        ):
            for operator in BINARY_OPERATORS:
                build(
                    current + [operator],
                    tokens_left - 1,
                    True,
                    open_parens,
                    0,  # Reset NOT depth after a binary operator
                    max_paren_depth,
                )

        # Add opening parenthesis if expecting a literal and within nesting limit
        if expect_literal and open_parens < max_paren_depth:
            build(
                current + [OPEN_PAREN],
                tokens_left - 1,
                True,
                open_parens + 1,
                0,  # Reset NOT depth after '('
                max_paren_depth,
            )

        # Add closing parenthesis if there are open ones and the last token is not an operator or open parenthesis
        if (
            open_parens > 0
            and not expect_literal
            and current
            and current[-1] not in ([OPEN_PAREN] + BINARY_OPERATORS + UNARY_OPERATORS)
        ):
            build(
                current + [CLOSE_PAREN],
                tokens_left - 1,
                False,
                open_parens - 1,
                0,  # Reset NOT depth after ')'
                max_paren_depth,
            )

    for length in range(1, n + 1):
        build([], length, True, 0, 0, max_paren_depth)

    logger.info(
        f"Generated {len(valid_expressions)} expressions in {time.time() - start_time:.2f} seconds."
    )
    return valid_expressions


def generate_error_expressions(valid_expressions, num_errors):
    """
    Generate unique erroneous expressions from valid ones.

    Args:
        valid_expressions (list): Valid Boolean logic expressions.
        num_errors (int): The total number of erroneous expressions to generate.

    Returns:
        list: A list of unique invalid Boolean logic expressions.
    """
    logger.info(f"Generating up to {num_errors} erroneous expressions.")

    errors = set()
    valid_expr_set = set(valid_expressions)
    max_attempts = num_errors * 5 if num_errors > 0 else 50
    attempts = 0

    def remove_token(expr):
        tokens = expr.split()
        if len(tokens) > 1:
            tokens.pop(random.randint(0, len(tokens) - 1))
        return " ".join(tokens)

    def replace_token(expr):
        tokens = expr.split()
        if tokens:
            idx = random.randint(0, len(tokens) - 1)
            if tokens[idx] in {"AND", "OR", "NOT"}:
                tokens[idx] = INVALID_TOKEN
            elif tokens[idx] in {"T", "F"}:
                tokens[idx] = "X"
        return " ".join(tokens)

    def shuffle_tokens(expr):
        tokens = expr.split()
        if len(tokens) > 2:
            random.shuffle(tokens)
        return " ".join(tokens)

    modification_functions = [remove_token, replace_token, shuffle_tokens]

    # Generate unique errors
    while len(errors) < num_errors and attempts < max_attempts:
        valid_expr = random.choice(valid_expressions)
        modification = random.choice(modification_functions)
        error_expr = modification(valid_expr)
        if error_expr not in valid_expr_set and error_expr not in errors:
            errors.add(error_expr)
        attempts += 1

    if len(errors) < num_errors:
        logger.warning(
            f"Requested {num_errors} errors but only generated {len(errors)}."
        )

    logger.info(f"Generated {len(errors)} erroneous expressions.")
    return list(errors)


def eval_expression(expression):
    safe_expr = (
        expression.replace("AND", "and")
        .replace("OR", "or")
        .replace("NOT", "not")
        .replace("T", "True")
        .replace("F", "False")
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            code_obj = compile(safe_expr, "<string>", "eval")

            result = eval(code_obj, {"__builtins__": None}, {})

            return result if isinstance(result, bool) else None
        except Exception as e:
            return None


def save_dataset(dataset, name):
    """
    Save the dataset to a file with the specified name.

    Args:
        dataset (list): The dataset containing logic expressions.
        name (str): The name (or path) of the file to save the dataset (without extension).
        errors (set): A set of invalid expressions. Defaults to None.
    """
    start_time = time.time()

    errors = set()

    logger.info(f"Saving dataset to {name}.txt with {len(dataset)} expressions.")

    combined_data = dataset + list(errors)

    directory = os.path.dirname(name)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Precompute all evaluations
    results = {}
    for expr in combined_data:
        if expr in errors:
            results[expr] = None
        else:
            results[expr] = eval_expression(expr)

    # Sort once before writing if sorting is desired
    combined_data.sort()

    with open(f"{name}.txt", "w") as f:
        for expr in combined_data:
            eval_result = results[expr]
            eval_string = (
                "T"
                if eval_result is True
                else ("F" if eval_result is False else ERROR_TOKEN)
            )
            f.write(f"{SOS_TOKEN} {expr} {EVAL_TOKEN} {eval_string} {EOS_TOKEN}\n")

    logger.info(f"Saved dataset in {time.time() - start_time:.2f} seconds.")


def load_dataset(data_dir="data", dataset_class_name="mt5"):
    """
    Load datasets from the project root first, and only search the boloco package if no matches are found.

    Args:
        data_dir (str): The directory where the datasets are saved.
        dataset_class_name (str): The class name of the dataset (e.g., mt5).

    Returns:
        tuple: (vocab, train_set, validate_set, test_set)
    """

    def load_set(set_name, base_dir):
        """
        Load a specific dataset split (train, validate, or test) from the given base directory.

        Args:
            set_name (str): The name of the set to load (e.g., 'train').
            base_dir (str): The base directory to search.

        Returns:
            list: A list of lines from the dataset file, or None if not found.
        """
        # Construct search pattern
        pattern = os.path.join(
            base_dir, dataset_class_name, "**", f"boloco-{set_name}-*.txt"
        )
        matching_files = glob.glob(pattern, recursive=True)

        if matching_files:
            # Use the first matching file
            with open(matching_files[0], "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return None

    def to_array(dataset):
        """
        Convert a dataset (list of lines) into a NumPy array of token indices.

        Args:
            dataset (list): The dataset containing lines of tokens.

        Returns:
            numpy.ndarray: The dataset as a NumPy array of token indices.
        """
        return np.array(
            [vocab[w] for line in dataset for w in line.split()],
            dtype=np.uint32,
        )

    # First search in the project directory
    project_dir = os.getcwd()
    train_set = load_set("train", project_dir)
    validate_set = load_set("validate", project_dir)
    test_set = load_set("test", project_dir)

    # If no files were found in the project directory, search in the package data
    if not (train_set and validate_set and test_set):
        from pkg_resources import resource_filename

        package_data_dir = resource_filename("boloco", data_dir)
        train_set = train_set or load_set("train", package_data_dir)
        validate_set = validate_set or load_set("validate", package_data_dir)
        test_set = test_set or load_set("test", package_data_dir)

    # If still missing any dataset, raise an error
    if not (train_set and validate_set and test_set):
        missing_sets = [
            name
            for name, dataset in zip(
                ["train", "validate", "test"], [train_set, validate_set, test_set]
            )
            if not dataset
        ]
        raise FileNotFoundError(
            f"Missing dataset files for the following sets: {', '.join(missing_sets)}."
        )

    # Build the vocabulary from all dataset splits
    vocab = set()
    for ds in [train_set, validate_set, test_set]:
        for expr in ds:
            vocab.update(expr.split())

    vocab = {token: idx for idx, token in enumerate(sorted(vocab))}

    # Convert datasets to arrays
    return vocab, to_array(train_set), to_array(validate_set), to_array(test_set)


def format_output_filename(
    max_tokens,
    seed,
    set_name,
    output_dir,
    expr_size,
    ratio,
    timestamp=time.strftime("%Y%m%d-%H%M%S"),
):
    dataset_class_name = f"mt{max_tokens}"
    return f"{output_dir}/{dataset_class_name}/{timestamp}/boloco-{set_name}-{dataset_class_name}_se{seed}_ex{expr_size}_ra{ratio*100:.0f}"


def split_dataset(valid_expressions, errors, train_ratio, validate_ratio):
    """
    Split the dataset into train, validate, and test sets with proportional errors.

    Args:
        valid_expressions (list): Valid Boolean logic expressions.
        errors (list): Erroneous expressions.
        train_ratio (float): Proportion of data in the training set.
        validate_ratio (float): Proportion of data in the validation set.

    Returns:
        tuple: (train_set, validate_set, test_set)
    """
    total_valid = len(valid_expressions)
    total_errors = len(errors)

    # Calculate sizes for each split
    train_valid_size = int(total_valid * train_ratio)
    validate_valid_size = int(total_valid * validate_ratio)
    test_valid_size = total_valid - train_valid_size - validate_valid_size

    train_error_size = int(total_errors * train_ratio)
    validate_error_size = int(total_errors * validate_ratio)
    test_error_size = total_errors - train_error_size - validate_error_size

    # Adjust for rounding discrepancies
    while train_valid_size + validate_valid_size + test_valid_size < total_valid:
        train_valid_size += 1
    while train_error_size + validate_error_size + test_error_size < total_errors:
        train_error_size += 1

    # Split valid expressions
    train_valid = valid_expressions[:train_valid_size]
    validate_valid = valid_expressions[
        train_valid_size : train_valid_size + validate_valid_size
    ]
    test_valid = valid_expressions[train_valid_size + validate_valid_size :]

    # Split errors
    train_errors = errors[:train_error_size]
    validate_errors = errors[train_error_size : train_error_size + validate_error_size]
    test_errors = errors[train_error_size + validate_error_size :]

    # Combine and shuffle each split
    train_set = train_valid + train_errors
    validate_set = validate_valid + validate_errors
    test_set = test_valid + test_errors

    random.shuffle(train_set)
    random.shuffle(validate_set)
    random.shuffle(test_set)

    return train_set, validate_set, test_set


def generate_data(args):
    """
    Generate the BoLoCo dataset according to the provided arguments.
    """
    logger.info("Starting BoLoCo dataset generation.")
    start_time = time.time()

    random.seed(args.seed)
    max_tokens = args.max_tokens
    logger.info(f"Generating expressions with max tokens: {max_tokens}")

    valid_expressions = list(generate_logic_expressions(max_tokens))
    total_expressions = len(valid_expressions)
    logger.info(f"Generated {total_expressions} valid expressions.")

    # Calculate total number of errors needed
    total_errors = int(total_expressions * args.error_ratio / (1 - args.error_ratio))
    logger.info(
        f"Generating {total_errors} erroneous expressions (error ratio: {args.error_ratio})."
    )
    errors = generate_error_expressions(valid_expressions, total_errors)

    # Adjust errors to ensure exact count
    while len(errors) < total_errors:
        errors.append(random.choice(errors))
    while len(errors) > total_errors:
        errors.pop()

    logger.info(f"Final error count: {len(errors)} (target: {total_errors})")

    # Split dataset
    train_set, validate_set, test_set = split_dataset(
        valid_expressions, errors, args.train_ratio, args.validate_ratio
    )

    # Log split sizes
    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Validate set size: {len(validate_set)}")
    logger.info(f"Test set size: {len(test_set)}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dataset(
        train_set,
        format_output_filename(
            max_tokens,
            args.seed,
            "train",
            args.dir,
            len(train_set),
            args.train_ratio,
            timestamp,
        ),
    )
    save_dataset(
        validate_set,
        format_output_filename(
            max_tokens,
            args.seed,
            "validate",
            args.dir,
            len(validate_set),
            args.validate_ratio,
            timestamp,
        ),
    )
    save_dataset(
        test_set,
        format_output_filename(
            max_tokens,
            args.seed,
            "test",
            args.dir,
            len(test_set),
            args.test_ratio,
            timestamp,
        ),
    )

    logger.info(
        f"BoLoCo dataset generation completed in {time.time() - start_time:.2f} seconds."
    )


def print_stats(args):
    """
    Calculate and display statistics for datasets in the specified directory.

    Args:
        args (Namespace): Command-line arguments containing the directory.
    """
    dataset_dir = args.dir

    if not os.path.exists(dataset_dir):
        logger.error(f"The directory {dataset_dir} does not exist.")
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    all_files = glob.glob(os.path.join(dataset_dir, "**", "*.txt"), recursive=True)

    if not all_files:
        logger.error(f"No dataset files found in directory: {dataset_dir}")
        return

    total_expressions = 0
    true_count = 0
    false_count = 0
    error_count = 0
    total_tokens = 0

    logger.info(f"Found {len(all_files)} files. Starting processing...")

    file_stats = []
    for file_path in all_files:
        file_true_count = 0
        file_false_count = 0
        file_error_count = 0
        file_total_expressions = 0
        file_total_tokens = 0

        logger.info(f"Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                file_total_expressions += 1
                total_expressions += 1
                tokens = line.split()
                file_total_tokens += len(tokens)
                total_tokens += len(tokens)

                # Check evaluation token results
                try:
                    if f"{EVAL_TOKEN} T" in line:
                        file_true_count += 1
                        true_count += 1
                    elif f"{EVAL_TOKEN} F" in line:
                        file_false_count += 1
                        false_count += 1
                    elif f"{EVAL_TOKEN} {ERROR_TOKEN}" in line:
                        file_error_count += 1
                        error_count += 1
                except Exception as e:
                    logger.warning(f"Error processing line: {line}. Reason: {e}")

        file_stats.append(
            {
                "file_path": file_path,
                "total_expressions": file_total_expressions,
                "true_count": file_true_count,
                "false_count": file_false_count,
                "error_count": file_error_count,
                "total_tokens": file_total_tokens,
            }
        )

    if total_expressions == 0:
        logger.warning("No valid expressions found in the dataset files.")
        return

    # Print individual file statistics
    print("\n--- Individual File Statistics ---")
    for stats in file_stats:
        file_true_percentage = (
            (stats["true_count"] / stats["total_expressions"] * 100)
            if stats["total_expressions"] > 0
            else 0
        )
        file_false_percentage = (
            (stats["false_count"] / stats["total_expressions"] * 100)
            if stats["total_expressions"] > 0
            else 0
        )
        file_error_percentage = (
            (stats["error_count"] / stats["total_expressions"] * 100)
            if stats["total_expressions"] > 0
            else 0
        )

        print(f"File: {stats['file_path']}")
        print(f"  Total expressions: {stats['total_expressions']}")
        print(
            f"  Expressions evaluating to T: {stats['true_count']} ({file_true_percentage:.2f}%)"
        )
        print(
            f"  Expressions evaluating to F: {stats['false_count']} ({file_false_percentage:.2f}%)"
        )
        print(
            f"  Erroneous expressions: {stats['error_count']} ({file_error_percentage:.2f}%)"
        )
        print(f"  Total tokens: {stats['total_tokens']}")

    # Print aggregate statistics
    true_percentage = (true_count / total_expressions) * 100
    false_percentage = (false_count / total_expressions) * 100
    error_percentage = (error_count / total_expressions) * 100

    print("\n--- Aggregate Statistics ---")
    print(f"{'Metric':<30}{'Count':<15}{'Percentage':<10}")
    print(f"{'Total expressions':<30}{total_expressions:<15}")
    print(f"{'Expressions evaluating to T':<30}{true_count:<15}{true_percentage:.2f}%")
    print(
        f"{'Expressions evaluating to F':<30}{false_count:<15}{false_percentage:.2f}%"
    )
    print(f"{'Erroneous expressions':<30}{error_count:<15}{error_percentage:.2f}%")
    print(f"{'Total tokens':<30}{total_tokens:<15}")


def main(args):
    if args.mode == "generate":
        generate_data(args)
    elif args.mode == "stats":
        print_stats(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Boolean logic expressions and split into BoLoCo datasets."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "stats"],
        default="generate",
        help="Mode to run the script in: 'generate' for dataset generation or 'stats' for statistics collection.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data",
        help="Directory to save the BoLoCo datasets in 'generate' mode or load them from in 'stats' mode.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5,
        help="Maximum number of tokens in the logic expression.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling the expressions.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Ratio of the training set."
    )
    parser.add_argument(
        "--validate_ratio",
        type=float,
        default=0.15,
        help="Ratio of the validation set.",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="Ratio of the test set."
    )
    parser.add_argument(
        "--error_ratio",
        type=float,
        default=0.05,
        help="Proportion of erroneous expressions to include in the dataset.",
    )

    args = parser.parse_args()
    main(args)
