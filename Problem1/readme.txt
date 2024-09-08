# **README.txt**

## **Project Overview**

This project is designed to enhance a Language Model (LLM) integrated within a Retrieval-Augmented Generation (RAG) framework. The primary goal is to improve the LLM's ability to accurately extract, summarize, and identify fee-related information from web content. The project uses a combined approach of Regular Expressions (Regex) for precise keyword detection and Direct Preference Optimization (DPO) for fine-tuning the model based on user feedback.


## **Requirements**

Ensure you have the following Python packages installed:

```bash
pip install llama-index transformers requests beautifulsoup4 openai
```

## **Usage Instructions**
### 1. **Using LLama Index**
running file stb_query.ipynb for an example
input openai key

### 2. **Fine-Tuning the LLM with DPO**

- **Input:** User feedback and generated outputs.
- **Script:** `dpo_finetuning.py`
- **Description:** This script fine-tunes the LLM using Direct Preference Optimization (DPO), incorporating user feedback to improve model performance.

```bash
python scripts/dpo_finetuning.py --feedback data/feedback/ --model models/pretrained/ --output models/finetuned/
```


## **Contributing**

Contributions to this project are welcome. Please ensure that any new scripts or modules follow the existing structure and are well-documented. Before submitting a pull request, run all scripts to ensure they work correctly with the existing setup.

## **License**

This project is licensed under the MIT License. See `LICENSE` for more details.

## **Contact**

For questions or support, please contact the project maintainer at [email@example.com].

---

This README provides an overview of the project, instructions on how to set it up, and details on how to run each script in the workflow. It ensures that anyone who reads it can understand and use the project effectively.