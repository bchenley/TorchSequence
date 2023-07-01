# TorchTimeSeries: PyTorch Time Series Analysis Package

TorchTimeSeries is a PyTorch-based package for time series analysis and modeling. It provides a flexible framework for applying machine learning techniques to time series data.

## Installation

To use TorchTimeSeries, follow these steps:

1. Create a virtual environment (optional but recommended): `python -m venv TorchTimeSeries-env`
2. Activate the virtual environment:
   - Windows: `.\TorchTimeSeries-env\Scripts\activate`
   - Linux/Mac: `source TorchTimeSeries-env/bin/activate`
3. Install the required dependencies: `pip install torch, pytorch_lightning`

## Getting Started

1. **Preparing Time Series Data**: To prepare time series data for TorchTimeSeries, ensure that your data is properly formatted and includes the necessary features and labels. Here are the steps to follow:

If your data is in the form of a dictionary, ensure that it includes the keys 'X' for input data, 'y' for output data, and 't' for the time variable. The 'X' key should correspond to the input features, the 'y' key should correspond to the output labels, and the 't' key should correspond to the time variable. Make sure that the values associated with each key are arrays or tensors containing the respective data.

Example dictionary format:

data = {
    'X': input_data,
    'y': output_data,
    't': time_data
}

If your data is in the form of a Pandas DataFrame, make sure that it includes columns with the names 'X' for input data, 'y' for output data, and 't' for the time variable. The 'X' column should contain the input features, the 'y' column should contain the output labels, and the 't' column should contain the time variable. Ensure that the values in each column are appropriately formatted.

Example DataFrame format:

df = pd.DataFrame({
    'X': input_data,
    'y': output_data,
    't': time_data
})

If you have your data stored in a CSV file, make sure that the file includes columns with the names 'X' for input data, 'y' for output data, and 't' for the time variable. The CSV file should contain the respective data in the corresponding columns. The file will be read in as a Pandas Dataframe and proceed from there.
   
2. **Model Development**: Build your time series model using the TorchTimeSeries framework. The package supports multi-input-multi-output, nonlinear, and nonstationary architectures. Customize your model's architecture by defining the necessary layers and leveraging PyTorch's modules.

## Training and Evaluation

1. **Training and Evaluation Procedures**: Configure the loss function, optimization algorithm, and performance metrics.
2. **PyTorch Lightning Integration**: Utilize PyTorch Lightning for seamless training, distributed training, and automatic optimization.
3. **Result Visualization**: Visualize the model's performance, interpretability, or predictions using TorchTimeSeries's tools.
4. **Saving Results**: Save trained models, evaluation metrics, and other artifacts for future reference.

## Contributing

Contributions to TorchTimeSeries are welcome! Please review the guidelines in [CONTRIBUTING.md](link-to-contributing.md) to get started.

## License

TorchTimeSeries is released under the [MIT License](https://github.com/bchenley/TorchTimeSeries/blob/main/LICENSE.txt).

## Contact Information

For inquiries or questions, please contact [henley.brandon@gmail.com].
