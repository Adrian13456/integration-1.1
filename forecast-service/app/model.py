"""Definition of the model and preprocessing pipeline for Ames Housing."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ames_train.csv"
)

# Internal feature names used by the model.
# These names match the API-to-model contract.
NUM = [
    "GrLivArea",
    "TotalBsmtSF",
    "GarageArea",
    "YearBuilt",
    "OverallQual",
]

CAT = ["MSZoning"]
TARGET = "SalePrice"

# Possible column name mappings (spaced -> normalized) for older CSV variants.
CSV_COLUMNS = {
    "Gr Liv Area": "GrLivArea",
    "Total Bsmt SF": "TotalBsmtSF",
    "Garage Area": "GarageArea",
    "Year Built": "YearBuilt",
    "Overall Qual": "OverallQual",
    "MS Zoning": "MSZoning",
    "SalePrice": "SalePrice",
}


def load_training_data() -> pd.DataFrame:
    """Load and normalize the Ames Housing training dataset.

    Supports both Kaggle-style column names (no spaces) and older
    variants with spaces.
    """
    df = pd.read_csv(DATA_PATH)

    # Remove accidental whitespace around column names.
    df.columns = df.columns.str.strip()

    # Rename spaced columns if present.
    rename_map = {k: v for k, v in CSV_COLUMNS.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    required = set(NUM + CAT + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    return df[list(required)].dropna(subset=[TARGET])


def train_model(df: pd.DataFrame | None = None) -> Pipeline:
    """Build and train the Ridge regression pipeline.

    Numerical features are imputed using the median and standardized.
    Categorical features are imputed using the most frequent value and
    one-hot encoded, with unknown categories ignored during inference.

    If *df* is provided, the model is fitted on that dataframe;
    otherwise the full training dataset is loaded.
    """
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                NUM,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                CAT,
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    if df is None:
        df = load_training_data()

    pipe.fit(
        df[NUM + CAT],
        df[TARGET],
    )

    return pipe
