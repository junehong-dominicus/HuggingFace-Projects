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
HuggingFace-Projects/
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
   git clone https://github.com/junehong-dominicus/HuggingFace-Projects.git
   cd HuggingFace-Projects
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

- **Text Generation**: GPT-2, LLaMA, Gemma
- **Embeddings**: BERT, RoBERTa
- **Multimodal**: BLIP-2, CLIP
- **Seq2Seq**: T5, BART

## 📊 Benchmarks

| Model      | Task              | Accuracy | Inference Time |
|------------|-------------------|----------|----------------|
| BERT       | Sentiment Analysis| 92.3%    | 0.12s/sample   |
| GPT-2      | Text Generation   | N/A      | 0.45s/sample   |

## 📚 Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Datasets Library](https://github.com/huggingface/datasets)

## 🤝 Contributing

Feel free to fork, star, and submit pull requests. Contributions are welcome!

## 📄 License

This project is licensed under the MIT License.
