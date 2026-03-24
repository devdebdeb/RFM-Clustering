import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging 
from src.config import CONFIG
from src.data_loader import load_data

logger = logging.getLogger(__name__)

def preprocess_data(config: CONFIG, df_raw: pd.DataFrame) -> pd.DataFrame:
    processed_df = df_raw.copy()
    
    processed_df[config.log_transform_features] = np.log1p(processed_df[config.log_transform_features])
    
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(processed_df[config.features])
    processed_df[config.features] = scaled_matrix
    
    return processed_df


def persist_data(config: CONFIG, processed_df: pd.DataFrame):
    path_object = config.full_processed_path

    path_object.parent.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(path_object, index=False)
    logger.info(f'Processed data persisted successfully at {path_object}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        config = CONFIG()
        df_raw = load_data(config)
        processed_data = preprocess_data(config, df_raw)
        persist_data(config, processed_data)
        logger.info('Smoke Test Passed: Features Engineered & Persisted')
    except Exception as e:
        logger.error(f'Smoke Test Failed: {e}')
        raise e