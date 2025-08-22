# Notebooks Directory

This directory contains Jupyter notebooks for ionic liquids property prediction using machine learning.

## Available Notebooks

### 1. ionic_liquids_simplified.ipynb
- **Purpose**: Quick start and simplified analysis
- **Recommended for**: Beginners, quick testing, educational purposes
- **Runtime**: ~10-30 minutes
- **Features**:
  - Basic data loading and preprocessing
  - Simple feature selection (top 6 features)
  - Model training with reduced optimization trials
  - Essential performance metrics
  - Basic visualization plots

**Key Sections**:
1. Data Loading and Exploration
2. Molecular Descriptor Generation
3. Feature Selection and Analysis
4. Model Training with Bayesian Optimization
5. Model Evaluation and Visualization
6. Results Interpretation

### 2. ionic_liquids_full.ipynb
- **Purpose**: Comprehensive analysis with advanced features
- **Recommended for**: Research, publication-quality results, advanced users
- **Runtime**: ~1-3 hours (depending on optimization parameters)
- **Features**:
  - Advanced feature importance analysis
  - Multiple feature selection strategies comparison
  - Nested cross-validation
  - Comprehensive model evaluation
  - SHAP interpretability analysis
  - Williams plot for outlier detection
  - Learning curve analysis
  - Residual analysis

**Key Sections**:
1. Data Loading and Quality Assessment
2. Comprehensive Feature Analysis
3. Feature Selection Strategy Comparison
4. Advanced Model Training and Optimization
5. Nested Cross-Validation
6. Model Interpretability (SHAP Analysis)
7. Outlier Detection (Williams Plot)
8. Performance Benchmarking
9. Results Export and Model Saving

## Running the Notebooks

### Prerequisites
Ensure you have installed all required packages:
```bash
pip install -r requirements.txt
```

### Getting Started
1. **Start with the simplified notebook** if you're new to the project
2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
3. **Navigate to the notebooks directory**
4. **Open the desired notebook**

### Execution Tips
- **Run cells sequentially**: Use Shift+Enter to execute cells in order
- **Read markdown cells**: Important explanations between code sections
- **Adjust parameters**: Modify hyperparameters in designated cells
- **Save regularly**: Use Ctrl+S to save your progress
- **Check outputs**: Verify results before proceeding to next sections

## Notebook Parameters

### Adjustable Parameters in Simplified Notebook
- `n_features`: Number of features to select (default: 6)
- `n_trials`: Optimization trials (default: 50)
- `test_size`: Train/test split ratio (default: 0.2)
- `random_state`: Random seed for reproducibility

### Adjustable Parameters in Full Notebook
- `n_features`: Number of features to select (default: 8)
- `n_trials`: Optimization trials (default: 200-680)
- `outer_cv`: Outer cross-validation folds (default: 10)
- `inner_cv`: Inner cross-validation folds (default: 5)
- `n_jobs`: Parallel processing cores (default: 4-18)

## Expected Outputs

### Generated Files
Both notebooks will create various output files:
- **Model files**: `*.joblib` (trained models)
- **Plots**: `*.png` (visualization results)
- **Data**: `filtered_data.csv` (preprocessed dataset)
- **Analysis**: JSON files with comprehensive results

### Visualization Outputs
- Scatter plots (predicted vs actual)
- Learning curves
- Feature importance plots
- Correlation heatmaps
- SHAP summary plots (full notebook)
- Williams plots (full notebook)
- Residual analysis plots (full notebook)

## Performance Expectations

### Typical Results
- **R² Score**: 0.80-0.92 (depending on dataset and parameters)
- **RMSE**: 0.12-0.22 (normalized scale)
- **Cross-validation stability**: ±0.06 standard deviation

### Runtime Estimates
- **Simplified notebook**: 10-30 minutes
- **Full notebook**: 1-3 hours
- **Time varies by**:
  - Dataset size
  - Number of optimization trials
  - Cross-validation complexity
  - Hardware specifications

## Troubleshooting

### Common Issues
1. **Kernel crashes**: Reduce n_trials or n_jobs parameters
2. **Memory errors**: Close other applications, reduce dataset size
3. **Import errors**: Reinstall required packages
4. **Long runtime**: Reduce optimization complexity for testing

### Performance Optimization
- **Use fewer trials** for initial testing (n_trials=20-50)
- **Adjust n_jobs** based on available CPU cores
- **Monitor memory usage** during execution
- **Save intermediate results** to prevent data loss

## Data Requirements

### Input Data Format
Both notebooks expect Excel files with:
- **Sheet name**: Specify in `load_data()` function if needed
- **Required columns**:
  - `smiles`: SMILES string representation
  - `ExtraPer`: Target property values
- **Optional columns**: Additional molecular properties

### Supported Datasets
- `Li_data.xlsx`: Lithium ionic liquids dataset
- `Eco1-mid.xlsx`: Eco-friendly ionic liquids (specify sheet: 'learn_regression')
- Custom datasets following the same format

## Customization Guide

### Adding New Features
1. Modify the `molecular_descriptors()` function
2. Add new RDKit descriptors or custom calculations
3. Update feature selection logic accordingly

### Changing Target Properties
1. Update the target column name in data loading
2. Adjust preprocessing if needed (scaling, transformation)
3. Modify evaluation metrics if appropriate

### Model Algorithm Changes
1. Replace ElasticNet with other sklearn regressors
2. Adjust hyperparameter search spaces
3. Update model evaluation sections

## Best Practices

### For Reproducible Results
- Set fixed random seeds throughout
- Document parameter choices
- Save model states and preprocessors
- Record software versions

### For Research Use
- Use the full notebook for comprehensive analysis
- Perform multiple runs with different random seeds
- Document all parameter choices and modifications
- Include uncertainty quantification

### For Educational Use
- Start with simplified notebook
- Follow markdown explanations carefully
- Experiment with different parameters
- Try different datasets

## Contact

For notebook-specific questions:
- **Author**: Kuo, Jui-An
- **Email**: lulala0208@gmail.com
- **Issues**: Submit via GitHub Issues