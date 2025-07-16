# Character-Level Language Model Training

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[![Lines of Code](https://img.shields.io/tokei/lines/github/simonpierreboucher02/LLM-Character-level-train-main)](https://github.com/simonpierreboucher02/LLM-Character-level-train-main)
[![Repo Size](https://img.shields.io/github/repo-size/simonpierreboucher02/LLM-Character-level-train-main)](https://github.com/simonpierreboucher02/LLM-Character-level-train-main)
[![Last Commit](https://img.shields.io/github/last-commit/simonpierreboucher02/LLM-Character-level-train-main)](https://github.com/simonpierreboucher02/LLM-Character-level-train-main)

A PyTorch implementation of a character-level GPT-style language model for text generation. This project demonstrates training a transformer-based model from scratch on character-level text data.

## 🚀 Features

- **Character-level tokenization**: Works directly with individual characters rather than subword tokens
- **GPT-style architecture**: Implements a transformer decoder with multi-head attention
- **Accelerated training**: Uses Hugging Face Accelerate for efficient training across devices
- **Text generation**: Includes inference capabilities for generating text from trained models
- **Modular design**: Clean separation of model architecture, training logic, and utilities

## 📁 Project Structure

```
LLM-Character-level-train-main/
├── model.py          # GPT model architecture implementation
├── train.py          # Training script with data loading and training loop
├── utils.py          # Utility functions for encoding/decoding and batching
└── README.md         # This file
```

## 🏗️ Model Architecture

The model implements a GPT-style transformer with the following components:

- **Token Embeddings**: Character-level embeddings
- **Positional Embeddings**: Learnable positional encodings
- **Transformer Blocks**: Multi-head self-attention + feed-forward networks
- **Layer Normalization**: Applied before each sub-layer
- **Language Model Head**: Linear projection to vocabulary size

### Model Parameters (Default Configuration)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Vocabulary Size** | Dynamic | Based on unique characters in training data |
| **Embedding Dimension** | 384 | Size of token and positional embeddings |
| **Number of Layers** | 6 | Number of transformer blocks |
| **Number of Attention Heads** | 6 | Multi-head attention configuration |
| **Block Size** | 256 | Context window length |
| **Dropout** | 0.2 | Dropout rate for regularization |
| **Batch Size** | 64 | Training batch size |
| **Learning Rate** | 3e-4 | AdamW optimizer learning rate |
| **Max Iterations** | 500 | Total training iterations |

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/simonpierreboucher02/LLM-Character-level-train-main.git
cd LLM-Character-level-train-main
```

2. Install required dependencies:
```bash
pip install torch accelerate
```

## 📚 Usage

### Training

1. Prepare your training data as a text file (e.g., `pg2554.txt`)

2. Run the training script:
```bash
python train.py
```

The training script will:
- Load and preprocess the text data
- Create character-level vocabulary
- Train the model for 500 iterations
- Save the trained model to `./scratchGPT/model.pt`
- Generate a sample text output

### Training Configuration

You can modify the training parameters in `train.py`:

```python
batch_size = 64
block_size = 256
max_iters = 500
learning_rate = 3e-4
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
```

### Model Usage

After training, you can load and use the model:

```python
from model import GPT
import torch

# Load the trained model
model = GPT(vocab_size, n_embed, block_size, n_layer, n_head, dropout)
model.load_state_dict(torch.load('./scratchGPT/model.pt'))

# Generate text
context = torch.tensor([[char2idx['H']]], dtype=torch.long)
generated_ids = model.generate(context, max_new_tokens=500)
generated_text = decode(generated_ids[0].tolist(), idx2char)
print(generated_text)
```

## 🔧 Key Components

### `model.py`
- `GPT`: Main model class with forward pass and generation methods
- `Block`: Transformer block with attention and feed-forward layers
- `MultiHeadAttention`: Multi-head self-attention mechanism
- `Head`: Single attention head implementation
- `FeedForward`: Position-wise feed-forward network

### `train.py`
- Data loading and preprocessing
- Training loop with evaluation
- Model saving and loading
- Text generation demonstration

### `utils.py`
- `encode()`: Convert text to token indices
- `decode()`: Convert token indices back to text
- `get_batch()`: Create training batches
- `estimate_loss()`: Evaluate model performance

## 🎯 Training Process

1. **Data Preparation**: Text is loaded and split into train/validation sets (90/10 split)
2. **Vocabulary Creation**: Unique characters are identified and mapped to indices
3. **Training Loop**: 
   - Batches are created from the training data
   - Forward pass computes loss
   - Backward pass updates model parameters
   - Evaluation is performed periodically
4. **Model Saving**: Trained model is saved for later use
5. **Generation**: Sample text is generated to demonstrate the model

## 📊 Performance & Metrics

### Model Statistics
- **Total Parameters**: ~1.2M parameters
- **Model Size**: ~5MB (saved model)
- **Training Time**: ~10-15 minutes (on CPU)
- **Memory Usage**: ~2GB RAM during training

### Key Features
- **Character-level tokenization**: No vocabulary size limits
- **Efficient training**: Uses Hugging Face Accelerate
- **Scalable architecture**: Easy to modify model size
- **Text generation**: Real-time inference capabilities

### Training Metrics
| Metric | Description |
|--------|-------------|
| **Loss Tracking** | Train/validation loss every 100 iterations |
| **Evaluation** | 200 evaluation iterations per checkpoint |
| **Model Checkpointing** | Automatic model saving |
| **Text Generation** | 500 token sample generation |

The model is designed for educational purposes and demonstrates:
- Character-level language modeling
- Transformer architecture implementation
- Training from scratch without pre-trained weights
- Text generation capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Simon Pierre Boucher**
- GitHub: [@simonpierreboucher02](https://github.com/simonpierreboucher02)
- ![GitHub followers](https://img.shields.io/github/followers/simonpierreboucher02?label=Followers&style=social)
- ![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher02?label=Stars&style=social)

## 📈 Repository Stats

![GitHub repo size](https://img.shields.io/github/repo-size/simonpierreboucher02/LLM-Character-level-train-main)
![GitHub language count](https://img.shields.io/github/languages/count/simonpierreboucher02/LLM-Character-level-train-main)
![GitHub top language](https://img.shields.io/github/languages/top/simonpierreboucher02/LLM-Character-level-train-main)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/simonpierreboucher02/LLM-Character-level-train-main)

## 🙏 Acknowledgments

This implementation is inspired by the original GPT architecture and serves as a learning resource for understanding transformer-based language models at the character level. 