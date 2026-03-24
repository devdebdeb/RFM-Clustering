import logging
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
from src.config import CONFIG  

logger = logging.getLogger(__name__)

class RFMSchema(pa.DataFrameModel):
    """
    Defines the expected structure of the input data.
    Acts as a gatekeeper: Bad data cannot pass this class.
    """
    APPLICATION_USER_ID: Series[int] = pa.Field(coerce=True, unique=True)

    R: Series[float] = pa.Field(ge=0, coerce=True)
    F: Series[float] = pa.Field(ge=0, coerce=True)
    M: Series[float] = pa.Field(ge=0, coerce=True)
    T: Series[float] = pa.Field(ge=0, coerce=True)

    class Config:
        strict = False

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Pandera contract to dataframe
    """
    try:
        validated_df = RFMSchema.validate(df, lazy=True)
        logger.info(f'Schema validation passed. Shape: {validated_df.shape}')
        return validated_df
    except pa.errors.SchemaErrors as e:
        logger.error('DATA INTEGRITY BREACH: Schema Validation Failed')
        logger.error(e.failure_cases)
        raise ValueError('Pipeline Aborted: Raw data violated RFM Schema.') from e

def load_data(config: CONFIG) -> pd.DataFrame:
    """
    Orchestrates the loading process: Read -> Validate -> Return.
    """
    path = config.full_raw_path
    logger.info(f'Initiating data ingestion from: {path}')

    if not path.exists():
        logger.error(f'File Not Found: {path}')
        raise FileNotFoundError(f'Missing raw data at {path}')
    
    try:
        df = pd.read_csv(path)
        df = validate_schema(df)
        return df

    except Exception as e:
        logger.critical(f'FATAL: Loader crashed. Reason {e}')
        raise e

if __name__ == '__main__':
    try:
        config = CONFIG()
        df = load_data(config)
        print('Smoke Test Passed: Data Loaded & Validated')
    except Exception as e:
        print(f'Smoke Test Failed: {e}')