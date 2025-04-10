# train.py
import logging

from utils import setup_logger
from pipeline import TrainingPipeline

class TrainOrchestrator:
    def __init__(self, data_path):
        self.logger = setup_logger("train_orchestrator")
        self.data_path = data_path
        self.pipeline = TrainingPipeline(self.data_path)

    def run_training(self):
        self.logger.info("🚀 Starting the Training Orchestration 🚀")

        self.pipeline.load_and_explore_data()
        if self.pipeline.data is not None:
            self.logger.info("✅ Data loaded and explored successfully.")
            self.pipeline.preprocess_data()
            if self.pipeline.X_train is not None:
                self.logger.info("✅ Data preprocessed successfully.")
                self.pipeline.initialize_models()
                self.logger.info("🛠️ Models initialized.")
                self.pipeline.train_and_tune_models()
                self.logger.info("🧪 Models trained and tuned.")
                self.pipeline.create_and_evaluate_ensembles()
                self.logger.info("ensemble created and evaluated.")
                self.pipeline.evaluate_models()
                self.logger.info("📊 Models evaluated.")
                self.pipeline.save_best_model()
                self.logger.info("💾 Best model saved successfully.")
                self.pipeline.save_all_models()  # Call the method to save all models
                self.logger.info("💾 All models saved successfully.")
            else:
                self.logger.warning("❌ Preprocessing did not complete successfully.")
        else:
            self.logger.warning("❌ Data loading failed, cannot proceed.")

        self.logger.info("🏁 Training Orchestration Finished 🏁")

if __name__ == "__main__":
    data_path = 'data/total_ksi.csv'
    orchestrator = TrainOrchestrator(data_path)
    orchestrator.run_training()