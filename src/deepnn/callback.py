from tensorflow.keras.callbacks import Callback
import json
import os

class CustomSaveCallback(Callback):
    def __init__(self, neural_network, interval=5):
        super().__init__()
        print(f"Setting saving callback every {interval} epochs")
        self.neural_network = neural_network
        self.interval = interval
        # Initialize the interim history with the model's metric names
        self.interim_history = {
            'epochs': [],
            'history': {metric: [] for metric in self.neural_network.model.metrics_names}
        }

    def on_epoch_end(self, epoch, logs=None):
        # Safely update interim history with the latest logs
        self.interim_history['epochs'].append(epoch)
        for key in self.interim_history['history'].keys():
            if logs.get(key) is not None:  # Check if the metric is in logs before appending
                self.interim_history['history'][key].append(logs[key])

        if (epoch + 1) % self.interval == 0:
            print(f"Epoch {epoch + 1}: Saving model and results...")
            self.neural_network.save_model(self.neural_network.instance_folder)
            
            # Update internal evaluation results
            self.neural_network.evaluate_model(verbose=0)
            
            # Get results using interim history
            results = self.neural_network.get_results(interim_results=self.interim_history)
            
            # Save results to JSON file
            results_path = os.path.join(self.neural_network.instance_folder, f'results_epoch_{epoch + 1}.json')
            with open(results_path, 'w') as f:
                json.dump(results, f)
            print(f"Results saved to {results_path}")

    def on_train_end(self, logs=None):
        # At the end of training, make sure the final history is recorded
        self.neural_network.history = self.interim_history
