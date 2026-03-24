from src.data_loader import load_data
from src.predict import predict_segments
from src.config import CONFIG
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def apply_label(config: CONFIG, df: pd.DataFrame):
    segment_map = {
        2: 'Champions',       # High Value ($69k), Frequent, Old
        1: 'Loyalists',       # Good Value ($5.5k), Frequent, Old
        3: 'At Risk',         # High Value ($17k), But Recency > 2.5 months (78 days)
        0: 'Hibernating',     # Low Value, Low Freq, Old (1898 days)
        4: 'Lost New'         # Low Value, Low Freq, New (427 days)
    }
    df['Segment'] = df['Cluster'].map(segment_map)

    logger.info('\n--- Final Segment Count ---')
    logger.info(df['Segment'].value_counts())
    logger.info('------------------------------\n')

    save_path = config.processed_data_dir / config.final_filename
    df.to_csv(save_path, index=False)
    logger.info(f'Final tagged database saved sucessfully at:\n{save_path}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        config = CONFIG()
        logger.info(f'Tagging customers using Best K={config.BEST_K}')
        df_tagged = predict_segments(config)
        apply_label(config, df_tagged)
    except Exception as e:
        logger.critical(f'Finalization Failed: {e}')