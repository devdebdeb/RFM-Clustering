from src.data_loader import load_data
from src.config import CONFIG
from src.features import preprocess_data
import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def predict_segments(config: CONFIG):
    k = config.BEST_K
    raw_df = load_data(config)
    processed_df = preprocess_data(config, raw_df)
    X = processed_df[config.features]

    model_path = config.model_dir / f'kmean_k{k}.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found at {model_path}. Run experiments first.')

    model = joblib.load(model_path)
    labels = model.predict(X)
    raw_df['Cluster'] = labels

    return raw_df

def generate_profile(config: CONFIG, df: pd.DataFrame) -> pd.DataFrame:
    profile = df.groupby('Cluster').agg({
        config.R: 'mean'
        ,config.F: 'mean'
        ,config.M: 'mean'
        ,config.T: 'mean'
        ,config.APPLICATION_USER_ID: 'count'
    }).round(2)
    return profile

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        config = CONFIG()
        df_tagged = predict_segments(config)
        logger.info('--- Cluster Profile Report ---')
        report = generate_profile(config, df_tagged)
        logger.info(f'\n{report}')
    
    except Exception as e:
        logger.critical(f'Prediction Failed: {e}')