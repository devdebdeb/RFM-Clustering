import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from src.config import CONFIG
from src.data_loader import load_data
from src.features import preprocess_data
import joblib

logger = logging.getLogger(__name__)

def plot_snake_chart(config: CONFIG):
    """
    Generates a Snake Plot to visualize the 4D cluster profiles.
    """

    k = config.BEST_K
    logger.info(f'Generating Snake Plot for K={k}')

    model_path = config.model_dir / f'kmean_k{k}.joblib'
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f'Model for K={k} missing. Run experiments first.')

    df_raw = load_data(config)
    df_scaled = preprocess_data(config, df_raw) 
    
    model = joblib.load(model_path)
    X = df_scaled[config.features] 
    df_scaled['Cluster'] = model.predict(X)

    df_melt = pd.melt(
        df_scaled.reset_index(), 
        id_vars=['Cluster'], 
        value_vars=config.features, 
        var_name='Metric', 
        value_name='Standardized Score'
    )

    plt.figure(figsize=(12, 6))
    
    sns.lineplot(
        data=df_melt, 
        x='Metric', 
        y='Standardized Score', 
        hue='Cluster', 
        palette='viridis', 
        marker='o',
        linewidth=2.5
    )
    
    plt.title(f'Cluster Segmentation Profile (K={k})')
    plt.xlabel('Features')
    plt.ylabel('Z-Score (Standardized Mean)')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Population Average')
    plt.legend(title='Cluster ID')
    plt.grid(True, alpha=0.3)
    
    output_path = config.processed_data_dir / f'snake_plot_k{k}.png'
    plt.savefig(output_path)
    logger.info(f"Snake plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        config = CONFIG()
        plot_snake_chart(config)
    except Exception as e:
        logger.critical(f'Visualization Failed: {e}')python 