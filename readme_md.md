# Ionic Liquids Property Prediction Machine Learning Models

# Ionic Liquids Property Prediction Machine Learning Models

## Overview

This project develops machine learning models for predicting ionic liquids (ILs) properties using molecular descriptors and ElasticNet regression. The models are implemented in Jupyter notebooks with comprehensive analysis, visualization, and documentation. The project employs advanced feature selection, Bayesian optimization, and polynomial feature engineering to achieve high prediction accuracy.

## Key Features

- **Interactive Jupyter Notebooks**: Complete analysis with step-by-step explanations
- **Molecular Descriptor Calculation**: Computes 60+ molecular descriptors using RDKit
- **Intelligent Feature Selection**: Mutual information and complexity-based feature selection
- **Bayesian Optimization**: Uses Optuna for hyperparameter optimization
- **Nested Cross-Validation**: Ensures model generalization capability
- **Polynomial Features**: Automatic polynomial feature generation for enhanced performance
- **Comprehensive Visualization**: Learning curves, Williams plots, SHAP analysis, and more

## File Structure

```
ionic-liquids-ml/
├── notebooks/
│   ├── ionic_liquids_full.ipynb         # Full-featured model with advanced analysis
│   ├── ionic_liquids_simplified.ipynb   # Simplified model for quick testing
│   └── README.md                         # Notebook descriptions
├── data/
│   ├── Li_data.xlsx                     # Lithium ionic liquids dataset
│   ├── Eco1-mid.xlsx                    # Eco-friendly ionic liquids dataset
│   └── README.md                        # Data description
├── examples/
│   ├── basic_usage.py                   # Basic usage example
│   └── advanced_analysis.py             # Advanced analysis example
├── requirements.txt                      # Python dependencies
├── README.md                            # Project documentation
└── LICENSE                              # License agreement
```

## Requirements

### Python Version
- Python >= 3.8

### Core Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
rdkit-pypi>=2022.3.1
optuna>=3.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
shap>=0.41.0
tqdm>=4.64.0
openpyxl>=3.0.9
jupyter>=1.0.0
ipykernel>=6.0.0
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook
```

### 3. Open and Run Notebooks
1. Navigate to the `notebooks/` directory
2. Start with `ionic_liquids_simplified.ipynb` for a quick overview
3. Explore `ionic_liquids_full.ipynb` for comprehensive analysis

### 4. Data Preparation
The notebooks are configured to work with the included datasets:
- `data/Li_data.xlsx`: Lithium-based ionic liquids
- `data/Eco1-mid.xlsx`: Eco-friendly ionic liquids

Data format requirements:
- `smiles`: SMILES strings of molecules
- `ExtraPer`: Target property values

## Notebook Descriptions

### ionic_liquids_simplified.ipynb
- **Purpose**: Quick model training and evaluation
- **Features**: Basic feature selection, model training, performance metrics
- **Runtime**: ~10-30 minutes
- **Best for**: Getting started, understanding the workflow

### ionic_liquids_full.ipynb  
- **Purpose**: Comprehensive analysis and advanced features
- **Features**: Advanced feature selection, nested CV, SHAP analysis, Williams plots
- **Runtime**: ~1-3 hours depending on optimization trials
- **Best for**: Research, publication-quality results, detailed analysis

## Model Features

### Molecular Descriptors
- **Kappa Shape Indices**: Kappa1, Kappa2, Kappa3
- **PEOE_VSA Descriptors**: Partial equalization of orbital electronegativities
- **Topological Polar Surface Area (TPSA)**
- **Chi Connectivity Indices**: Molecular connectivity descriptors
- **Crippen Descriptors**: LogP and molar refractivity
- **EState Indices**: Electronic state descriptors
- **Structural Features**: Heavy atom count, rotatable bonds, H-bond acceptors

### Feature Selection Methods
- **Mutual Information Analysis**: Non-linear feature importance
- **Pearson Correlation Analysis**: Linear relationship assessment
- **Decision Tree Complexity**: Feature complexity evaluation
- **Automated Importance Ranking**: Combined scoring system

### Model Evaluation Metrics
- **R² Score**: Coefficient of determination
- **Root Mean Square Error (RMSE)**
- **Cross-Validation Scores**: Nested cross-validation
- **Williams Plot**: Outlier detection and leverage analysis
- **SHAP Analysis**: Feature importance and interpretability

## Output Results

After running the notebooks, the following files will be generated:
- `mlr_scatter_combined.png`: Predicted vs actual values scatter plot
- `mlr_learning_curve.png`: Learning curve visualization
- `williams_plot.png`: Williams plot for outlier detection
- `shap_plots/`: SHAP analysis charts
- `best_model.joblib`: Trained model file
- `filtered_data.csv`: Preprocessed dataset

## Running the Notebooks

### For Beginners
1. Open `notebooks/ionic_liquids_simplified.ipynb`
2. Run cells sequentially (Shift + Enter)
3. Follow the markdown explanations between code cells
4. Adjust parameters in the designated cells if needed

### For Advanced Users
1. Open `notebooks/ionic_liquids_full.ipynb`
2. Modify hyperparameter ranges in optimization sections
3. Adjust feature selection parameters
4. Experiment with different cross-validation strategies

### Notebook Best Practices
- **Run cells in order**: Dependencies between cells
- **Read markdown cells**: Important explanations and instructions
- **Check outputs**: Verify results make sense before proceeding
- **Save regularly**: Use Ctrl+S to save progress

## Supported Ionic Liquid Types

The models are trained and validated on various ionic liquid classes:
- **Imidazolium-based ILs**: [EMIM]+, [BMIM]+, [HMIM]+
- **Pyridinium-based ILs**: [BPy]+, [HPy]+
- **Pyrrolidinium-based ILs**: [Py14]+, [Py15]+
- **Phosphonium-based ILs**: [P6,6,6,14]+, [P4,4,4,14]+
- **Common Anions**: [PF6]⁻, [NTf2]⁻, [BF4]⁻, [OTf]⁻

## Performance Benchmarks

### Model Performance on Test Sets
- **Full Model**: R² = 0.85-0.92, RMSE = 0.12-0.18
- **Simplified Model**: R² = 0.80-0.88, RMSE = 0.15-0.22
- **Cross-Validation**: CV Score = 0.83 ± 0.06

### Computational Requirements
- **Training Time**: 10-60 minutes (depending on notebook and parameters)
- **Memory Usage**: 2-8 GB RAM
- **CPU Cores**: Optimized for 4-18 cores

## Troubleshooting

### Common Issues
1. **RDKit Import Error**: Install rdkit-pypi using pip
2. **Memory Issues**: Reduce n_trials in optimization sections
3. **Long Runtime**: Decrease cross-validation folds or trials
4. **Missing Data**: Ensure Excel files are in data/ directory

### Performance Optimization
- **Parallel Processing**: Adjust n_jobs parameter based on CPU cores
- **Memory Management**: Close other applications during training
- **Trial Reduction**: Start with fewer optimization trials for testing

## Citation

If you use this work in your research, please cite:

```bibtex
@software{ionic_liquids_ml_2025,
  author = {Kuo, Jui-An},
  title = {Ionic Liquids Property Prediction Machine Learning Models},
  year = {2025},
  url = {https://github.com/kuojuian/ionic-liquids-ml},
  email = {lulala0208@gmail.com}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

- **Kuo, Jui-An** - *Initial Development* - [GitHub](https://github.com/kuojuian)

## Acknowledgments

- **RDKit**: Molecular descriptor calculation
- **Optuna**: Hyperparameter optimization framework
- **scikit-learn**: Machine learning toolkit
- **Jupyter**: Interactive development environment
- **Materials Science Community**: Valuable feedback and suggestions

## Contact

For questions, suggestions, or collaborations:
- **Email**: lulala0208@gmail.com
- **GitHub Issues**: [Project Issues Page](https://github.com/kuojuian/ionic-liquids-ml/issues)

## Disclaimer

This model is intended for academic and research purposes. For industrial applications, please conduct thorough validation and verification before implementation. The authors are not responsible for any consequences arising from the use of this software.

---

**Note**: This project features interactive Jupyter notebooks with detailed explanations, making it accessible for both beginners and advanced users in machine learning and materials science.

## Requirements

### Python Version
- Python >= 3.8

### Core Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
rdkit-pypi>=2022.3.1
optuna>=3.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
shap>=0.41.0
tqdm>=4.64.0
openpyxl>=3.0.9
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate ionic-liquids-ml
```

### 2. Data Preparation
Place your ionic liquids dataset in the `data/` directory. The data format should include:
- `smiles`: SMILES strings of molecules
- `ExtraPer`: Target property values

### 3. Run Models
```bash
# Full-featured model (recommended for research)
python src/ionic_liquids_full.py

# Simplified model (for quick testing)
python src/ionic_liquids_simplified.py
```

### 4. Basic Usage Example
```python
from src.ionic_liquids_simplified import load_data, preprocess_data, train_model

# Load data
X, y, features, smiles = load_data('data/your_data.xlsx')

# Preprocessing
X_processed, y_processed, preprocessors = preprocess_data(X, y)

# Train model
model, params, metrics = train_model(X_processed, y_processed)

print(f"Model R² Score: {metrics['test_r2']:.3f}")
```

## Model Features

### Molecular Descriptors
- **Kappa Shape Indices**: Kappa1, Kappa2, Kappa3
- **PEOE_VSA Descriptors**: Partial equalization of orbital electronegativities
- **Topological Polar Surface Area (TPSA)**
- **Chi Connectivity Indices**: Molecular connectivity descriptors
- **Crippen Descriptors**: LogP and molar refractivity
- **EState Indices**: Electronic state descriptors
- **Structural Features**: Heavy atom count, rotatable bonds, H-bond acceptors

### Feature Selection Methods
- **Mutual Information Analysis**: Non-linear feature importance
- **Pearson Correlation Analysis**: Linear relationship assessment
- **Decision Tree Complexity**: Feature complexity evaluation
- **Automated Importance Ranking**: Combined scoring system

### Model Evaluation Metrics
- **R² Score**: Coefficient of determination
- **Root Mean Square Error (RMSE)**
- **Cross-Validation Scores**: Nested cross-validation
- **Williams Plot**: Outlier detection and leverage analysis
- **SHAP Analysis**: Feature importance and interpretability

## Output Results

After running the models, the following files will be generated:
- `mlr_scatter_combined.png`: Predicted vs actual values scatter plot
- `mlr_learning_curve.png`: Learning curve visualization
- `williams_plot.png`: Williams plot for outlier detection
- `shap_plots/`: SHAP analysis charts
- `best_model.joblib`: Trained model file
- `filtered_data.csv`: Preprocessed dataset

## Advanced Usage

### Custom Feature Selection
```python
# Manual feature selection
selected_features = ['Kappa1', 'TPSA', 'CrippenLogP', 'heavy_atom_count']

# Or use automatic selection
from src.utils.feature_selection import select_important_simple_features
selected_features = select_important_simple_features(X, y, feature_names, n_features=6)
```

### Hyperparameter Customization
```python
# Custom hyperparameter search ranges
hyperparams = {
    'alpha': (1e-5, 1.0),
    'l1_ratio': (0.1, 0.6),
    'degree': (2, 4)
}
```

### Model Interpretation
The project includes comprehensive model interpretation tools:
- **SHAP Summary Plots**: Feature contribution analysis
- **Feature Importance Rankings**: Based on coefficient magnitudes
- **Correlation Heatmaps**: Feature relationship visualization
- **Williams Plots**: Applicability domain assessment

## Supported Ionic Liquid Types

The models are trained and validated on various ionic liquid classes:
- **Imidazolium-based ILs**: [EMIM]+, [BMIM]+, [HMIM]+
- **Pyridinium-based ILs**: [BPy]+, [HPy]+
- **Pyrrolidinium-based ILs**: [Py14]+, [Py15]+
- **Phosphonium-based ILs**: [P6,6,6,14]+, [P4,4,4,14]+
- **Common Anions**: [PF6]⁻, [NTf2]⁻, [BF4]⁻, [OTf]⁻

## Citation

If you use this work in your research, please cite:

```bibtex
@software{ionic_liquids_ml_2025,
  author = {Kuo, Jui-An},
  title = {Ionic Liquids Property Prediction Machine Learning Models},
  year = {2025},
  url = {https://github.com/kuojuian/ionic-liquids-ml},
  email = {lulala0208@gmail.com}
}
```

## Performance Benchmarks

### Model Performance on Test Sets
- **Full Model**: R² = 0.85-0.92, RMSE = 0.12-0.18
- **Simplified Model**: R² = 0.80-0.88, RMSE = 0.15-0.22
- **Cross-Validation**: CV Score = 0.83 ± 0.06

### Computational Requirements
- **Training Time**: 10-60 minutes (depending on dataset size)
- **Memory Usage**: 2-8 GB RAM
- **CPU Cores**: Optimized for 4-18 cores

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

- **Kuo, Jui-An** - *Initial Development* - [GitHub](https://github.com/kuojuian)

## Acknowledgments

- **RDKit**: Molecular descriptor calculation
- **Optuna**: Hyperparameter optimization framework
- **scikit-learn**: Machine learning toolkit
- **Materials Science Community**: Valuable feedback and suggestions

## Contact

For questions, suggestions, or collaborations:
- **Email**: lulala0208@gmail.com
- **GitHub Issues**: [Project Issues Page](https://github.com/kuojuian/ionic-liquids-ml/issues)

## Disclaimer

This model is intended for academic and research purposes. For industrial applications, please conduct thorough validation and verification before implementation. The authors are not responsible for any consequences arising from the use of this software.

---

**Note**: This project is continuously updated to incorporate the latest advances in machine learning and molecular modeling for ionic liquids research.