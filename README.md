# 🤗 Hugging Face Model Playground

Welcome to the Hugging Face Model Playground! This repository is dedicated to studying, experimenting with, and benchmarking transformer models using the Hugging Face ecosystem.

## 📌 Objectives

- Explore various model architectures (e.g., BERT, GPT, T5, Vision Transformers)
- Run inference and fine-tuning examples
- Benchmark performance across tasks and hardware
- Document optimization techniques (quantization, mixed precision, etc.)
- Share deployment-ready scripts and notebooks

## 🧰 Repository Structure

```
huggingface-models-study/
│
├── notebooks/              # Jupyter notebooks for experiments
├── scripts/                # Python scripts for training/inference
├── benchmarks/             # Performance metrics and comparisons
├── datasets/               # Custom or downloaded datasets
├── docs/                   # Documentation and guides
└── README.md               # This file
```

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/huggingface-models-study.git
   cd huggingface-models-study
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run your first notebook:
   ```bash
   jupyter notebook notebooks/text-generation-gpt2.ipynb
   ```

## 🧪 Model Categories

| Category         | Example Models        | Use Cases                            |
|------------------|-----------------------|--------------------------------------|
| Text Generation  | GPT-2, LLaMA, Gemma   | Chatbots, story generation, prompts  |
| Text Classification | BERT, RoBERTa      | Sentiment analysis, spam detection   |
| Multimodal       | CLIP, BLIP-2          | Image captioning, text-to-image      |
| Embeddings       | Sentence-BERT, MiniLM | Semantic search, clustering          |
| Seq2Seq          | T5, BART              | Translation, summarization           |

## 📊 Benchmarks

| Model      | Task              | Accuracy | Inference Time |
|------------|-------------------|----------|----------------|
| BERT       | Sentiment Analysis| 92.3%    | 0.12s/sample   |
| GPT-2      | Text Generation   | N/A      | 0.45s/sample   |

## 📓 Example: Sentiment Classification with BERT

This module demonstrates how to fine-tune BERT for binary sentiment classification using the IMDb dataset.

### 🔧 Setup

```bash
pip install transformers datasets scikit-learn
```

### 📁 Files

- `notebooks/bert_sentiment.ipynb`: Interactive training and evaluation
- `scripts/train_bert_sentiment.py`: CLI-based training script
- `benchmarks/bert_sentiment.json`: Accuracy, F1 score, inference time

### 📊 Results

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 92.3%     |
| F1 Score   | 91.8%     |
| Inference  | 0.12s/sample |

### 📌 Notes

- Uses `AutoModelForSequenceClassification` from Hugging Face
- Includes early stopping and learning rate scheduling
- Compatible with CPU and GPU

## 📚 Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Datasets Library](https://github.com/huggingface/datasets)

## 🤝 Contributing

Feel free to fork, star, and submit pull requests. Contributions are welcome!

## 📄 License

This project is licensed under the MIT License.
