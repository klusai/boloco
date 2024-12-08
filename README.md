# BoLoCo: Boolean Logic Expression Generator

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

## Contributing
To contribute to BoLoCo:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/MyFeature`).
3. Commit your changes (`git commit -m 'Add MyFeature'`).
4. Push the branch (`git push origin feature/MyFeature`).
5. Open a pull request detailing your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for further information.