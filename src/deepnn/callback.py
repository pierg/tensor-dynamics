from tensorflow.keras.callbacks import Callback
import json
import os

from src.shared.utils import save_dict_to_json_file


class CustomSaveCallback(Callback):
    def __init__(self, neural_network, interval=5):
        super().__init__()
        print(f"Setting saving callback every {interval} epochs")
        self.neural_network = neural_network
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print(f"Epoch {epoch + 1}: Saving model and results...")
            self.neural_network.save_model(self.neural_network.instance_folder)

            # Update internal evaluation results
            self.neural_network.evaluate_model(verbose=0)

            # Get results using interim history
            results = self.neural_network.get_results(interim=True)

            # Save results to JSON file

            save_dict_to_json_file(
                results,
                self.neural_network.instance_folder / f"results_epoch_{epoch + 1}.json",
            )
