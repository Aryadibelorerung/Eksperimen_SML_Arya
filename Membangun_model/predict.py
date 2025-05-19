import mlflow

model_uri = 'runs:/292e217fd7184dbbb0919b44fe1062f9/model'

# Replace INPUT_EXAMPLE with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
input_data = {
    "umur": 28,
    "penghasilan": 6000,
    "status": "lajang"
}

# Verify the model with the provided input data using the logged dependencies.
# For more details, refer to:
# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
    env_manager="uv",
)