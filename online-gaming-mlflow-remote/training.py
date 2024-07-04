from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # noqa
from sklearn.preprocessing import StandardScaler  # noqa
from sklearn.preprocessing import TargetEncoder  # noqa
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from plots import (
    create_confusion_matrices,
    create_learning_curves,
    plot_feature_importance,
)
from reports import make_classification_report_frame

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5555")

mlflow.set_experiment("Online Gaming Behavior - Engagement Level Prediction")
mlflow.set_experiment_tags(
    {
        "project": "Online Gaming Behavior",
        "task": "Classification",
    }
)
run = mlflow.start_run()

online_gaming_behavior_dataset = pd.read_csv("training_data.csv")

features = online_gaming_behavior_dataset.drop(columns=["PlayerID", "EngagementLevel"])
target = online_gaming_behavior_dataset["EngagementLevel"]

split_test_size = 0.2
split_random_state = 42

mlflow.log_param("split_test_size", split_test_size)
mlflow.log_param("split_random_state", split_random_state)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=split_test_size, random_state=split_random_state
)


mlflow.log_param("x_train_shape", X_train.shape)
mlflow.log_param("x_test_shape", X_test.shape)
mlflow.log_param("y_train_shape", y_train.shape)
mlflow.log_param("y_test_shape", y_test.shape)


# List features
categorical_features = ["Gender", "Location", "GameGenre", "GameDifficulty"]
discrete_variables = ["Age", "SessionsPerWeek", "AvgSessionDurationMinutes", "AchievementsUnlocked"]
continuous_variable = ["PlayTimeHours"]


# Preprocess the data
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

transformers = [
    # ('one-hot-encoder', OneHotEncoder(sparse_output=False), categorical_features),
    # ('target-encoder', TargetEncoder(), categorical_features),
    ("scaler", StandardScaler(), continuous_variable + discrete_variables)
]
preprocessor = ColumnTransformer(
    transformers,
    verbose_feature_names_out=False,
)
preprocessor.set_output(transform="pandas")

X_train_prep = preprocessor.fit_transform(X_train, y_train)
X_test_prep = preprocessor.transform(X_test)


X_train_prep = pd.DataFrame(X_train_prep, columns=preprocessor.get_feature_names_out())
X_test_prep = pd.DataFrame(X_test_prep, columns=preprocessor.get_feature_names_out())

X_test_prep.to_csv(output_dir / "X_test_prep.csv", index=False)
X_train_prep.to_csv(output_dir / "X_train_prep.csv", index=False)

mlflow.log_artifact(output_dir / "X_test_prep.csv")

#################
# Train the model
#################
xgb_model = XGBClassifier()

xgb_model.fit(X_train_prep, y_train)

mlflow.log_params({
    f"xgb_{param}": value for param, value in xgb_model.get_params().items()
})


#################
# Save the model and auxiliary objects
#################

joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
mlflow.log_artifact(output_dir / "label_encoder.pkl")
joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
mlflow.log_artifact(output_dir / "preprocessor.pkl")
joblib.dump(xgb_model, output_dir / "xgb_model.pkl")
mlflow.log_artifact(output_dir / "xgb_model.pkl")

signature = infer_signature(X_train_prep, xgb_model.predict(X_train_prep))

mlflow.xgboost.log_model(xgb_model, "xgb_model", signature=signature)

model_name = "online-gaming-engagement-level-prediction"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/xgb_model", 
    name=model_name
)


#################
# Evaluate the model
#################

y_pred_test = xgb_model.predict(X_test_prep)
y_pred_train = xgb_model.predict(X_train_prep)

test_accuracy = accuracy_score(y_test, y_pred_test)
train_accuracy = accuracy_score(y_train, y_pred_train)

mlflow.log_metric("train_accuracy", train_accuracy)
mlflow.log_metric("test_accuracy", test_accuracy)

# Print classification report
clf_report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
for label in label_encoder.classes_:
    for metric in clf_report[label]:
        mlflow.log_metric(f"{metric}_{label}", clf_report[label][metric])

report_frame = make_classification_report_frame(clf_report, label_encoder.classes_)

mlflow.log_table(report_frame, "classification_report.json")

# Log feature importances
feature_importance = pd.DataFrame(
    {"feature": X_train_prep.columns, "importance": xgb_model.feature_importances_}
).sort_values("importance", ascending=False)

mlflow.log_table(feature_importance, "feature_importance.json")

feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

feature_importance_fig = plot_feature_importance(feature_importance)
mlflow.log_figure(feature_importance_fig, "feature_importance.png")

# Create confusion matrices for train and test
confusion_matrix_figure = create_confusion_matrices(label_encoder, y_train, y_pred_train, y_test, y_pred_test)
mlflow.log_figure(confusion_matrix_figure, "confusion_matrix.png")

# Create a learning curve plot
learning_curve_figure = create_learning_curves(xgb_model, X_train_prep, y_train)

mlflow.log_figure(learning_curve_figure, "learning_curve.png")

print(f"Model version: {model_version.name} {model_version.version}")
print(f"Experiment ID: {run.info.experiment_id}")
print(f"Run ID: {run.info.run_id}")
