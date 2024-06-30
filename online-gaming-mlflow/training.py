from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # noqa
from sklearn.preprocessing import StandardScaler  # noqa
from sklearn.preprocessing import TargetEncoder  # noqa
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from ydata_profiling import ProfileReport

from plots import (
    create_confusion_matrices,
    create_learning_curves,
    plot_feature_importance,
)

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

online_gaming_behavior_dataset = pd.read_csv("online_gaming_behavior_dataset.csv")

data_report = ProfileReport(online_gaming_behavior_dataset, title="Data Report")
data_report.to_file(output_dir / "data_report.html")

features = online_gaming_behavior_dataset.drop(columns=["PlayerID", "EngagementLevel"])
target = online_gaming_behavior_dataset["EngagementLevel"]

split_test_size = 0.2
split_random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=split_test_size, random_state=split_random_state
)


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

#################
# Train the model
#################
xgb_model = XGBClassifier()

xgb_model.fit(X_train_prep, y_train)


#################
# Save the model and auxiliary objects
#################
joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
joblib.dump(xgb_model, output_dir / "xgb_model.pkl")

#################
# Evaluate the model
#################

y_pred_test = xgb_model.predict(X_test_prep)
y_pred_train = xgb_model.predict(X_train_prep)

test_accuracy = accuracy_score(y_test, y_pred_test)
train_accuracy = accuracy_score(y_train, y_pred_train)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Print classification report
clf_report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
for label in label_encoder.classes_:
    for metric in clf_report[label]:
        print(f"Class {label} ({metric}): {clf_report[label][metric]}")


# Log feature importances
feature_importance = pd.DataFrame(
    {"feature": X_train_prep.columns, "importance": xgb_model.feature_importances_}
).sort_values("importance", ascending=False)

feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

feature_importance_fig = plot_feature_importance(feature_importance)
feature_importance_fig.savefig(output_dir / "feature_importance.png")

# Create confusion matrices for train and test
confusion_matrix_figure = create_confusion_matrices(label_encoder, y_train, y_pred_train, y_test, y_pred_test)
confusion_matrix_figure.savefig(output_dir / "confusion_matrix.png")

# Create a learning curve plot
learning_curve_figure = create_learning_curves(xgb_model, X_train_prep, y_train)

learning_curve_figure.savefig(output_dir / "learning_curve.png")
