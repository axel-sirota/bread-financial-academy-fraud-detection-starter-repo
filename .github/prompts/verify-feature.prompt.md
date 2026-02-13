# Feature Engineering Code Verification Prompt

You are reviewing feature engineering code to ensure it follows the project's feature engineering standards defined in `features.instructions.md`.

## Verification Checklist

### Function Naming Convention
- [ ] Feature functions start with `create_` or `extract_`
- [ ] Column names in the DataFrame are descriptive (e.g., `amount_log`, `is_weekend`, NOT `f1`)
- [ ] No cryptic or abbreviated column names

### Function Pattern Requirements
All feature functions MUST satisfy these requirements:

1. **DataFrame Parameter**
   - [ ] First parameter is a `pd.DataFrame`
   - [ ] Parameter is properly type-hinted

2. **Return Value**
   - [ ] Returns a NEW DataFrame (never modifies input)
   - [ ] Uses `.copy()` to prevent in-place modifications
   - [ ] Return type is properly annotated as `-> pd.DataFrame`

3. **Column Validation**
   - [ ] Function validates that required input columns exist
   - [ ] Raises appropriate exception if required columns are missing
   - [ ] Validation happens before data processing

4. **Documentation**
   - [ ] Includes Google-style docstring
   - [ ] Docstring contains brief description
   - [ ] Docstring includes `Args:` section with column requirements
   - [ ] Docstring includes `Returns:` section describing output
   - [ ] Docstring includes `Raises:` section if applicable
   - [ ] Docstring includes usage example

### Code Quality
- [ ] No direct DataFrame mutations (e.g., `df['new_col'] = ...`)
- [ ] All operations done on copy before returning
- [ ] Proper type hints for all parameters and returns
- [ ] Follows snake_case naming convention
- [ ] No print() statements (uses logging if needed)

## Example of Compliant Code

```python
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features.

    Extracts and engineers time-based features from timestamp columns
    to identify patterns in transaction timing.

    Args:
        df: DataFrame with 'hour' and 'day_of_week' columns

    Returns:
        DataFrame with new time features added:
        - is_weekend: Binary flag for weekend transactions

    Raises:
        KeyError: If required columns are missing
    """
    required_cols = ['hour', 'day_of_week']
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Missing required columns: {required_cols}")
    
    result = df.copy()
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    return result
```

## Common Issues to Flag

- ❌ `df['new_col'] = ...` (direct mutation)
- ❌ Function names like `feature1()` or `extract_f1()`
- ❌ Missing or incomplete docstrings
- ❌ No input validation
- ❌ Not returning a new DataFrame
- ❌ Missing type hints
- ❌ Cryptic column names after transformation

## Review Process

1. Check function naming follows `create_`/`extract_` pattern
2. Verify DataFrame is copied before modification
3. Validate all required columns are checked
4. Review docstring completeness
5. Confirm no in-place mutations occur
6. Verify type hints are present and correct
7. Check column naming is descriptive
