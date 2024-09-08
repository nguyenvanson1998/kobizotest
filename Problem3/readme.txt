Here's a basic `README.txt` for your project:

---

# Multi-Label Classification of Solidity Smart Contracts

## Project Overview

This project is designed to perform multi-label classification on Solidity smart contract code using a pre-trained T5 model. The model identifies various vulnerabilities or characteristics within the smart contracts, such as `Reentrancy Attack`, `Unsafe Delegate calls`, `Phishing Through tx.origin`, etc.

### Key Features:
- **Data Input**: Smart contract code in Solidity.
- **Model**: A T5-based encoder model fine-tuned for classifying smart contracts.
- **Output**: Multi-label classification indicating the presence of various vulnerabilities.

## Prerequisites

### Python Version
- Python 3.6 or higher

### Required Libraries
- `torch`: PyTorch library for deep learning.
- `transformers`: Hugging Face's Transformers library.
- `scikit-learn`: For metrics and data splitting.
- `pandas`: For data manipulation and reading Excel files.
- `openpyxl`: For reading Excel files in Pandas.

### Installing Dependencies

To install all required libraries, run the following command:

```bash
pip install -r requirements.txt
```

Ensure that you have a GPU available for faster training. However, the code can also be executed on a CPU, albeit more slowly.

## Project Structure

- **`P3 - sample dataset.xlsx`**: The Excel file containing the smart contract code and corresponding labels.
- **`requirements.txt`**: Contains the list of Python libraries required to run the project.
- **`train_model.py`**: The script that loads the data, trains the model, and evaluates its performance.

## How to Run the Project

1. **Prepare the Dataset**: Ensure the Excel file containing the dataset is in the same directory as the script. The Excel file should have:
   - `source code` column with Solidity code.
   - Binary label columns for each vulnerability (e.g., `Reentrancy Attack`, `Unsafe Delegate calls`, etc.).

2. **Install Dependencies**: Make sure all dependencies are installed by running:
   
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Training Script**:
   
   Execute the training script to train the model:
   
   ```bash
   python train_model.py
   ```

4. **Evaluate the Model**: After training, the model will output accuracy and F1 score metrics, which can be used to evaluate its performance on the test set.

## Customization

- **Modify Hyperparameters**: You can adjust the batch size, learning rate, number of epochs, and other hyperparameters directly in the `train_model.py` script.

- **Change the Model**: If you want to experiment with other models, you can replace the `T5EncoderModel` with another encoder model from the Hugging Face library.

## Notes

- **Data Format**: The dataset must be correctly formatted with a `source code` column for the input text and separate columns for each label.
- **Performance**: The model's performance can vary based on the quality and quantity of the data provided.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

This `README.txt` provides an overview, setup instructions, and basic usage information for your project. Feel free to adjust it to better fit any additional specifics or instructions you have for the project.