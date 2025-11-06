# Industrial Equipment Monitoring

Lightweight Kedro-based project that provides data ingestion, processing, model training and reporting utilities for industrial equipment anomaly detection.

Change parameters and hyperparameters in [`parameters.yml`](conf/base/parameters.yml) if needed

## Estrutura do Projeto

- **`conf/`** - Project settings
    - **`base/`** - Global settings
        - [`catalog.yml`](conf/base/catalog.yml) - Catalog all datasets
        - [`parameters.yml`](conf/base/parameters.yml) - Store parameters and hyperparameters
    - **`local/`** - Local settings
        - `credentials.yml`- Store credentials
- **`logs`** - Store log files
- **`data/`** - Organize datasets
    - **`01_raw/`** - Raw data
    - **`02_intermediate/`** - Intermediate data
    - **`03_primary/`** - Clean data
    - **`04_feature/`** - ML features
    - **`05_model_input/`** - Data ready for training
    - **`06_models/`** - Trained models
    - **`07_model_output/`** - Model outputs/predictions
    - **`08_reporting/`** - Data for reports/dashboards
- **`src/`** - Source code
    - [`main.py`](src/main.py) - Arquivo principal
    - [`hooks.py`](src/hooks.py) - Hooks for logger
    - [`pipeline_registry.py`](src/pipeline_registry.py) - Register the project's pipelines
    - [`settings.py`](src/settings.py) - Project settings
    - **`utils/`** - Utility functions
        - [`get_root_path.py`](src/utils/get_root_path.py) - Get the project's root path
        - [`logging_config.py`](src/utils/logging_config.py) - Personalized logger
    - **`pipelines/`** - Pipelines
        - **`data_processing/`** - Clean and process raw data
            - [`nodes.py`](src/industrial_equipment_monitoring/pipelines/data_processing/nodes.py) - Pipeline node to clean and split into training/test
            - [`pipeline.py`](src/industrial_equipment_monitoring/pipelines/data_processing/pipeline.py) - Setup nodes to execute pipeline
        - **`model_training/`** - Optimize and train model
            - [`get_xgb_params.py`](src/industrial_equipment_monitoring/pipelines/model_training/get_xgb_params.py) - Get parameters and hyperparameters from parameters.yml
            - [`optimize`](src/industrial_equipment_monitoring/pipelines/model_training/optimize.py) - Obtain best hyperparameters from an Optuna study
            [`nodes.py`](src/industrial_equipment_monitoring/pipelines/model_training/nodes.py) - Pipeline node for model training
            - [`pipeline.py`](src/industrial_equipment_monitoring/pipelines/model_training/pipeline.py) - Setup nodes to execute pipeline
        - **`reporting/`** - Calculate thresholds, make predictions and create graphs
            - **`functions/`** - Functions to calculate metrics and graphs
                - [`plots.py`](src/industrial_equipment_monitoring/pipelines/reporting/functions/plots.py) - Create graphs for the model
                - [`predictions.py`](src/industrial_equipment_monitoring/pipelines/reporting/functions/prediction.py) - Calculate thresholds, make predictions and calculate metrics
            - [`nodes.py`](src/industrial_equipment_monitoring/pipelines/reporting/nodes.py) - Pipeline node for reporting data
            - [`pipeline.py`](src/industrial_equipment_monitoring/pipelines/reporting/pipeline.py) - Setup nodes to execute pipeline
- **`tests/`** - Project tests
    - [`test_logger.py`](tests/test_logger.py) - Test personalized logger
    - **`pipelines/`** - Test for pipelines
        - **`data_processing/`**
            - [`test_pipeline.py`](tests/pipelines/data_processing/test_pipeline.py)
        - **`model_training/`** - Clean and process raw data
            - [`test_pipeline.py`](tests/pipelines/model_training/test_pipeline.py)
        - **`reporting/`** - Clean and process raw data
            - [`test_pipeline.py`](tests/pipelines/model_training/test_pipeline.py)
- [`pyproject.toml`](pyproject.toml) - Single standardized file that defines the entire project configuration
- [`README.md`](README.md) - This file
- **`readme/`** - Files to use in README.md
- [`requirements.txt`](requirements.txt) - Required libraries, it's optional since pyproject.toml already has it
- [`uv.lock`](uv.lock) - Freeze exact versions of all project dependencies to ensure reproducible and consistent installations.

## How to run
1. Install [`uv`](https://docs.astral.sh/uv/) — An extremely fast Python package and project manager, written in Rust.

2. Install python 3.13
```
uv python install 3.13
```

3. Clone repository
```
git clone https://github.com/Rafael-soares-oliveira/industrial_equipment_monitoring
```

4. Acess directory
```
cd industrial_equipment_monitoring
```

5. Create a virtual environment with python 3.13
```
uv venv
```

6. Activate venv
```
.venv/Scripts/activate -> Windows
source .venv/bin/activate -> Linux/Mac
```

7. Install all required libraries
```
uv sync
```

8. Execute the pipeline
```
kedro run
```

## Pipelines

### data_processing
- Clean and split data into training/test
- Save as `train_data` and `test_data` in [`catalog.yml`](conf/base/catalog.yml)

## model_training
- Optimize hyperparameters using Optuna
- Train model using `XGBoost.Booster`, the native API
- Save as `final_model` and `best_params` in [`catalog.yml`](conf/base/catalog.yml)

## reporting
- Calculate optimal threshold using `threshold_method` in [`parameters.yml`](conf/base/parameters.yml)
- Make predictions using `defaul threshold` and `optimal threshold`
- Calculate metrics
- Create graphs of the model
- Save as `predictions`, `report_data`, `confusion_matrix_plot`, `threshold_analysis_plot`, `roc_curve_plot`, `precision_recall_curve_plot`, `feature_importance_plot` in [`parameters.yml`](conf/base/parameters.yml)


## Pending Issues
- Create tests for the pipelines
- Create a node to generate a report in PDF
- Register a CLI command to create a connection with Tabpy for deploy the model in a Tableau dashboard