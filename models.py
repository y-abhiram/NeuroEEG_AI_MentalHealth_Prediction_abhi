from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers
import os

# Load old model
old_model = load_model("models/cnn_feature_extractor.h5", compile=False)

# Rebuild new model
new_model = Sequential()

for i, layer in enumerate(old_model.layers):
    config = layer.get_config()

    # Remove problematic keys
    config.pop("batch_input_shape", None)
    config.pop("batch_shape", None)

    # For first layer only, set input_shape manually
    if i == 0:
        config["input_shape"] = (8, 1)  # ✅ Replace with your correct input shape

    # Recreate the layer without batch_shape
    new_layer = layer.__class__.from_config(config)
    new_model.add(new_layer)

# Set weights
new_model.set_weights(old_model.get_weights())

# Save clean model
os.makedirs("models", exist_ok=True)
new_model.save("models/cnn_final_cleaned_model.h5")
print("✅ Model rebuilt and saved as cnn_final_cleaned_model.h5")

