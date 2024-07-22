import os
import traceback
import logging
from tensorflow.keras.models import load_model
from icu_watch_package.preprocessor import preprocess_input
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_trained_model(model_name='sepsis_model2.keras'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', model_name)

    if True:
        logging.info(f"Attempting to load model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")

        # Print model summary for verification
        model.summary()

        return model

    #except FileNotFoundError as e:
    #    logging.error(f"Model file not found: {str(e)}")
    #    return None
    #except Exception as e:
    #    logging.error(f"Error loading the model: {str(e)}")
    #    logging.debug(traceback.format_exc())
    #    return None

if __name__ == "__main__":
    # Test the function
    model = load_trained_model()
    if model is not None:
        logging.info("Model loaded successfully in test run.")
    else:
        logging.error("Failed to load model in test run.")
