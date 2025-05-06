# Code Completion Prototype with Code LLaMA

This repository contains a prototype implementation for code completion using the **Code LLaMA** large language model (LLM). The goal is to provide accurate and context-aware code completions to improve developer productivity and code quality.

## Features

* **Context-Aware Code Completions**: Generates relevant code suggestions based on existing code context.
* **Evaluation Metrics**: Clearly defined metrics (BLEU and ROUGE scores) to evaluate the accuracy and efficiency of generated completions.
* **Incremental Development**: Project designed to be expanded step-by-step, starting from basic evaluation and metrics to advanced integrations.
* **Interactive UI**: Streamlit-based interface to interactively test and visualize code completions.

## Technologies Used

* Python 3.x
* Code LLaMA model (LLaMA family)
* PyTorch
* Hugging Face Transformers
* Streamlit

## Installation

Clone the repository:

```bash
git clone CodeCompletionPrototype
cd code-completion-prototype
```

Create and activate a conda environment:

```bash
conda create -n codecompletion python=3.10 -y
conda activate codecompletion
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: If running on Apple Silicon (M1/M2), ensure you install the PyTorch version optimized for your hardware:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Running the Prototype

To run the basic code completion inference:

```bash
python src/inference.py --model_path <path_to_model> --input_code "your code snippet here"
```

### Evaluating the Model

To run evaluations using predefined metrics (BLEU and ROUGE):

```bash
python src/evaluate.py --test_dataset path/to/test_dataset
```

### Launching Streamlit App

Run the Streamlit app for an interactive experience:

```bash
streamlit run src/app.py
```

## Project Structure

```
.
├── src
│   ├── app.py             # Streamlit UI for interactive code completions
│   ├── inference.py       
│   ├── train.py
│   ├── evaluation.py
│   ├── error_analysis.py    
│   └── config.yaml           
├── requirements.txt       # Python dependencies
└── README.md              # This documentation
```

## Roadmap

* Implement advanced context management
* Fine-tune model on domain-specific datasets
* Integrate interactive IDE plugins

## Contributing

Pull requests and suggestions are encouraged! Please open an issue first to discuss significant changes.