# Model Training Code Verification Prompt

You are reviewing model training code to ensure it follows the project's model training standards defined in `model.instructions.md`.

## MLflow Integration Requirements

ALL training functions MUST satisfy these requirements:

### 1. Experiment Setup
- [ ] Function calls `mlflow.set_experiment(experiment_name)` with appropriate name
- [ ] Experiment is set BEFORE starting a run
- [ ] Experiment name is descriptive (e.g., "fraud-detection")

### 2. Parameter Logging
- [ ] `mlflow.log_params()` is called to log all hyperparameters
- [ ] Called within `mlflow.start_run()` context
- [ ] All model parameters are included (max_depth, learning_rate, n_estimators, etc.)
- [ ] Additional params logged: train size, test size, number of features

### 3. Metrics Logging
- [ ] `mlflow.log_metrics()` is called with ALL required metrics
- [ ] Required metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- [ ] Metrics are correctly calculated using sklearn
- [ ] Called within `mlflow.start_run()` context

### 4. Model Logging
- [ ] `mlflow.xgboost.log_model()` is called to persist the model
- [ ] Model is logged with artifact path (e.g., "model")
- [ ] Called within `mlflow.start_run()` context
- [ ] Enables model reloading for inference and serving

## Code Structure Requirements

### Function Naming and Documentation
- [ ] Training function has descriptive name starting with `train_` (e.g., `train_fraud_model`)
- [ ] Function has Google-style docstring with:
  - Brief description of what it trains
  - `Args:` section documenting all parameters
  - `Returns:` section describing the returned model
  - `Raises:` section if applicable

### Type Hints
- [ ] All function parameters have type hints
- [ ] Return type is annotated (e.g., `-> xgb.XGBClassifier`)
- [ ] Uses proper types: `pd.DataFrame`, `pd.Series`, `Dict`, `Optional`, etc.

### Data Handling
- [ ] Proper train/test split using `train_test_split`
- [ ] Stratified split for imbalanced classification: `stratify=y`
- [ ] `random_state` set for reproducibility
- [ ] Feature/target preparation is handled correctly

## Common Issues to Flag

- ❌ `mlflow.set_experiment()` not called
- ❌ Parameters logged outside of `mlflow.start_run()` context
- ❌ Missing any required metric (accuracy, precision, recall, f1, roc_auc)
- ❌ Model not logged with `mlflow.xgboost.log_model()`
- ❌ Non-stratified train/test split for imbalanced data
- ❌ Missing type hints on function parameters
- ❌ Incomplete docstring
- ❌ No error handling for invalid inputs

## Example of Compliant Code

```python
def train_fraud_model(
    df: pd.DataFrame,
    target_col: str = 'is_fraud',
    experiment_name: str = 'fraud-detection',
    test_size: float = 0.2,
    run_name: Optional[str] = None,
    **model_params
) -> xgb.XGBClassifier:
    """Train fraud detection model with MLflow tracking.

    Trains an XGBoost classifier on fraud detection data with full
    MLflow experiment tracking for reproducibility and model comparison.

    Args:
        df: DataFrame with features and target column.
        target_col: Name of the target column (default: 'is_fraud').
        experiment_name: MLflow experiment name for grouping runs.
        test_size: Fraction of data to use for testing (default: 0.2).
        run_name: Optional name for this specific run.
        **model_params: XGBoost hyperparameters to override defaults.

    Returns:
        Trained XGBoost classifier ready for inference.

    Raises:
        ValueError: If required columns are missing from DataFrame.
        KeyError: If target column not found in DataFrame.
    """
    params = {**DEFAULT_PARAMS, **model_params}

    # Set experiment BEFORE starting run
    mlflow.set_experiment(experiment_name)

    X, y, feature_cols = prepare_features(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=params.get('random_state', 42),
        stratify=y  # Critical for imbalanced data
    )

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(params)

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate and log metrics
        metrics = {
            'accuracy': accuracy_score(y_test, model.predict(X_test)),
            'precision': precision_score(y_test, model.predict(X_test)),
            'recall': recall_score(y_test, model.predict(X_test)),
            'f1': f1_score(y_test, model.predict(X_test)),
            'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }
        mlflow.log_metrics(metrics)

        # Log model for serving
        mlflow.xgboost.log_model(model, 'model')

    return model
```

## Review Checklist

### Before Merging
1. [ ] All MLflow requirements satisfied
2. [ ] All five required metrics computed and logged
3. [ ] Type hints complete for all parameters and returns
4. [ ] Google-style docstring present and comprehensive
5. [ ] Train/test split is stratified
6. [ ] No direct DataFrame mutations
7. [ ] Proper error handling present
8. [ ] Logging instead of print statements
9. [ ] Code tested with sample data
10. [ ] Notebook refactor successfully translated to module

## Notebook Refactor Verification

When refactoring from notebook (00_fraud_detection_pipeline.ipynb):
- [ ] All feature engineering logic preserved from notebook cells
- [ ] Model training logic matches notebook implementation
- [ ] All metrics computed in same way as notebook
- [ ] Hyperparameters match notebook defaults
- [ ] MLflow logging enhanced beyond notebook scope
- [ ] Function signature is clean and documented
- [ ] Error handling added where notebook might fail
