import logging
import warnings
from src.config import CONFIG
from src.features import preprocess_data, persist_data
from src.data_loader import load_data
from src.finalize_segments import apply_label
from src.predict import predict_segments

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
warnings.simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger("RFM_Orchestrator")

def run_production_pipeline():
    """
    Executes the full RFM Inference Pipeline:
    Ingest -> Feature Engineering -> Tagging -> Report
    """
    logger.info("--- STARTING PRODUCTION PIPELINE ---")
    config = CONFIG()

    try:
        # 1. Ingest & Feature Engineering
        logger.info("STEP 1: Ingesting and Engineering Features...")
        df_raw = load_data(config)
        df_processed = preprocess_data(config, df_raw)
        persist_data(config, df_processed)

        # 2. Inference (Tagging)
        logger.info(f"STEP 2: Tagging Customers using Model K={config.BEST_K}...")
        df_tagged = predict_segments(config)

        # 3. Finalization (Translation to Business Labels)
        logger.info("STEP 3: Generating Final Segment Report...")
        apply_label(config, df_tagged)

        logger.info("--- PIPELINE SUCCESS ---")
        logger.info(f"Deliverable available at: {config.processed_data_dir / config.final_filename}")

    except Exception as e:
        logger.critical(f"--- PIPELINE FAILED: {e} ---")
        raise e

if __name__ == "__main__":
    run_production_pipeline()