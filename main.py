
import logging
from toxicpred.pipeline.train_pipeline import TrainPipeline
import warnings
warnings.filterwarnings("ignore")

def main():
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__=="__main__":
    main()