from src.config import CONFIG
from src.data_loader import load_data
from src.features import preprocess_data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)

def run_experiment(config: CONFIG, processed_data: pd.DataFrame):
    results = {
        'k_value': []
        ,'inertia_score': []
        ,'silhouette_score': []
    }
    for k in config.K_RANGE:
        #1. Train
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init='auto')
        kmeans.fit(processed_data[config.features])

        #2. Score
        inertia = kmeans.inertia_
        labels = kmeans.labels_
        silhouette = silhouette_score(processed_data[config.features], labels)

        #3. Record Metrics
        results['k_value'].append(k)
        results['inertia_score'].append(inertia)
        results['silhouette_score'].append(silhouette)
    
        #4. Persistence
        model_path = config.model_dir / f'kmean_k{k}.joblib'
        joblib.dump(kmeans, model_path)
        logger.info(f'Model saved to {model_path.name}')

    return pd.DataFrame(results)

def plot_results(results: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        results['k_value']
         ,results['inertia_score']
          ,marker='o'
          ,color='blue'
          ,label='Inertia'
    )
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Elbow)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.bar(
        results['k_value']
        ,results['silhouette_score']
        ,alpha=0.3
        ,color='red'
        ,label='Silhouette'
    )
    ax2.set_ylabel('Silhouette Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('K-Means Optimization: Inertia vs Silhouette')
    plt.grid(True, alpha=0.3)

    output_path = config.processed_data_dir / 'experiment_results.png'
    plt.savefig(output_path)
    logger.info(f'Plot saved to {output_path}')
    plt.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = CONFIG()
    
    try:
        logger.info('1. Load Data & Process')
        raw_df = load_data(config)
        processed_df = preprocess_data(config, raw_df)
        
        logger.info(f'3. Running Experiments on K={config.K_RANGE}...')
        results_df = run_experiment(config, processed_df)

        metrics_path = config.processed_data_dir / 'experiment_metrics.csv'
        results_df.to_csv(metrics_path, index=False)
        logger.info(f'Metrics saved to {metrics_path}')

        logger.info('4. Plotting Results...')
        plot_results(results_df)
    except Exception as e:
        logger.critical(f'Experiment Failed: {e}')
        raise e