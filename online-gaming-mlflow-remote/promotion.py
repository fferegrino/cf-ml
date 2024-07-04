import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5555")

client = MlflowClient()


model_name = "online-gaming-engagement-level-prediction"


import pandas as pd

holdout = pd.read_csv("holdout.csv")
y_true = holdout["EngagementLevel"]


challenger_model_version = None
champion_model_version = None

try:
    challenger_model_version = client.get_model_version_by_alias(
        model_name, "challenger"
    )
except:
    pass

try:
    champion_model_version = client.get_model_version_by_alias(
        model_name, "champion"
    )
except:
    pass


from pathlib import Path
import joblib

def download_artifacts(run_id):
    destination = Path("artifacts", run_id)
    destination.mkdir(parents=True, exist_ok=True)

    preprocessor_path = client.download_artifacts(
        run_id=run_id,
        path="preprocessor.pkl",
        dst_path=destination ,
    )

    label_encoder_path = client.download_artifacts(
        run_id=run_id,
        path="label_encoder.pkl",
        dst_path=destination,
    )

    model = mlflow.xgboost.load_model(
        f"runs:/{run_id}/xgb_model",
        dst_path=destination,
    )

    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    return preprocessor, label_encoder, model

    


from sklearn.metrics import accuracy_score

if challenger_model_version and champion_model_version:
    print("Descargando artefactos del modelo challenger y champion")
    challenger_preprocessor, challenger_label_encoder, model = download_artifacts(challenger_model_version.run_id)
    champion_preprocessor, champion_label_encoder, champion_model = download_artifacts(champion_model_version.run_id)

    X_challenger = challenger_preprocessor.transform(holdout)
    y_challenger_pred = model.predict(X_challenger)

    X_champion = champion_preprocessor.transform(holdout)
    y_champion_pred = champion_model.predict(X_champion)


    # Calculating accuracy
    challenger_accuracy = accuracy_score(
        challenger_label_encoder.inverse_transform(y_challenger_pred),
        y_true,
    )

    champion_accuracy = accuracy_score(
        champion_label_encoder.inverse_transform(y_champion_pred),
        y_true,
    )


    if challenger_accuracy > champion_accuracy:
        print("El modelo challenger es mejor")

        print("Marcando el modelo champion como archived")
        client.set_model_version_tag(
            champion_model_version.name,
            champion_model_version.version,
            "archived",
            "true",
        )

        print("Promoviendo el modelo challenger a champion")
        client.set_registered_model_alias(
            challenger_model_version.name,
            "champion",
            challenger_model_version.version,
        )
        client.delete_registered_model_alias(
            champion_model_version.name,
            "champion",
        )
    else:
        print("Champion model is better: demoting challenger model to archived")
        print("El modelo champion es mejor: archivando el modelo challenger")

        print("Demoting challenger model to archived")
        client.set_model_version_tag(
            challenger_model_version.name,
            challenger_model_version.version,
            "archived",
            "true",
        )

else:
    # TODO: Implementa la lógica para cuando no hay retador, modelo campeón o ninguno de los dos.
    pass



