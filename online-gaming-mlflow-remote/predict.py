import mlflow
from mlflow.tracking import MlflowClient
import tempfile
import joblib
import pandas as pd
import sys

mlflow.set_tracking_uri("http://localhost:5555")


client = MlflowClient()
model_name = "online-gaming-engagement-level-prediction"

version = client.get_model_version_by_alias(model_name, "champion")

with tempfile.TemporaryDirectory() as temp_dir:

    preprocessor_path = client.download_artifacts(
        run_id=version.run_id,
        path="preprocessor.pkl",
        dst_path=temp_dir ,
    )

    label_encoder_path = client.download_artifacts(
        run_id=version.run_id,
        path="label_encoder.pkl",
        dst_path=temp_dir ,
    )

    model = mlflow.xgboost.load_model(
        f"runs:/{version.run_id}/xgb_model",
        dst_path=temp_dir,
    )

    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    
input_data = pd.read_csv(sys.argv[1])
output_file = sys.argv[2]

X = preprocessor.transform(input_data)
y_pred = model.predict(X)

y_pred = label_encoder.inverse_transform(y_pred)

predictions = pd.DataFrame(y_pred, columns=["EngagementLevel"]).to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")


