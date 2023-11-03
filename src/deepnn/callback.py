from tensorflow.keras.callbacks import Callback
import json
import os

class CustomSaveCallback(Callback):
    def __init__(self, neural_network, interval=5):
        super().__init__()
        self.neural_network = neural_network
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print(f"Epoch {epoch + 1}: Saving model and results...")
            self.neural_network.save_model(self.neural_network.instance_folder)
            
            # Update internal evaluation results
            self.neural_network.evaluate_model(verbose=0)
            results = self.neural_network.get_results()
            
            # Save results to JSON file
            results_path = os.path.join(self.neural_network.instance_folder, f'results_epoch_{epoch + 1}.json')
            with open(results_path, 'w') as f:
                json.dump(results, f)
            print(f"Results saved to {results_path}")

