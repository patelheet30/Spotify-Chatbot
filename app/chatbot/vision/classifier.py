import logging
import os
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from keras.api.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)
from keras.api.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.api.models import load_model
from keras.api.preprocessing import image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

logger = logging.getLogger(__name__)


class MusicAlbumClassifier:
    def __init__(self, models_dir: str = "output"):
        self.models_dir = models_dir
        self.models = {}
        self.class_names = ["Classical Era", "Digital Era"]
        self.model_img_size = {"efficientnet": (224, 224), "resnet": (224, 224)}

        self._load_models()

    def _load_models(self) -> None:
        loaded_count = 0

        try:
            efficientnet_path = os.path.join(
                self.models_dir, "efficientnet_final_model.keras"
            )
            if os.path.exists(efficientnet_path):
                self.models["efficientnet"] = load_model(efficientnet_path)
                logger.info("EfficientNet model loaded successfully.")
                loaded_count += 1
            else:
                logger.warning(
                    f"EfficientNet model file not found at {efficientnet_path}"
                )
        except Exception as e:
            logger.error(f"Error loading EfficientNet model: {e}")

        try:
            resnet_path = os.path.join(self.models_dir, "resnet_final_model.keras")
            if os.path.exists(resnet_path):
                self.models["resnet"] = load_model(resnet_path)
                logger.info("ResNet model loaded successfully.")
                loaded_count += 1
            else:
                logger.warning(f"ResNet model file not found at {resnet_path}")
        except Exception as e:
            logger.error(f"Error loading ResNet model: {e}")

        if not self.models:
            logger.error(
                "No models were loaded. Please check the model paths and file formats."
            )
        else:
            logger.info(f"Loaded {loaded_count} models for music album classification")
            logger.info(f"Available models: {', '.join(self.models.keys())}")

    def preprocess_image(self, img_path: str, model_name: str) -> np.ndarray:
        try:
            img_size = self.model_img_size.get(model_name, (224, 224))
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            if model_name == "efficientnet":
                return efficientnet_preprocess(img_array)
            elif model_name == "resnet":
                return resnet_preprocess(img_array)  # type: ignore
            else:
                return img_array / 255.0
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def classify_image(
        self, img_path: str, ensemble: bool = False
    ) -> Dict[str, Union[str, float]]:
        if not self.models:
            return {"era": "Unknown", "confidence": 0.0, "error": "No models available"}

        try:
            results = {}
            available_models = list(self.models.keys())

            if not ensemble and "resnet" in self.models:
                models_to_use = ["resnet"]
            elif ensemble and len(self.models) > 1:
                models_to_use = available_models
            else:
                models_to_use = available_models
                if ensemble and len(self.models) == 1:
                    logger.warning(
                        f"Ensemble requested but only one model available: {available_models[0]}"
                    )

            logger.info(f"Using models for classification: {', '.join(models_to_use)}")

            for model_name in models_to_use:
                model = self.models[model_name]
                preprocessed_img = self.preprocess_image(img_path, model_name)
                predictions = model.predict(preprocessed_img, verbose=0)

                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                era = self.class_names[predicted_class_idx]

                results[model_name] = {"era": era, "confidence": confidence}

            if len(models_to_use) == 1 or not ensemble:
                model_name = models_to_use[0]
                result = results[model_name]
                result["model_used"] = model_name
                return result

            eras_votes = {}
            for model_name, result in results.items():
                era = result["era"]
                confidence = result["confidence"]

                if era not in eras_votes:
                    eras_votes[era] = 0

                eras_votes[era] += confidence

            final_era = max(eras_votes, key=eras_votes.get)  # type: ignore
            total_confidence = sum(eras_votes.values())
            final_confidence = (
                eras_votes[final_era] / total_confidence if total_confidence > 0 else 0
            )

            return {
                "era": final_era,
                "confidence": final_confidence,
                "model_results": results,  # type: ignore
                "model_used": "ensemble",
            }

        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return {"era": "Unknown", "confidence": 0.0, "error": str(e)}
