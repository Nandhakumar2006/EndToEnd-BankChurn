import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def run_models(models, X_train, X_test, y_train, y_test,
               X_train_res=None, y_train_res=None):
    """
    Trains and logs multiple models in MLflow.
    Automatically selects SMOTE or original data for each model.
    """

    smote_models = ["XGBoost", "Gradient Boosting", "Decision Tree", "Logistic Regression"]

    for name, model in models.items():
        print(f"\nTraining model: {name}")

        if name in smote_models and X_train_res is not None and y_train_res is not None:
            X_used, y_used = X_train_res, y_train_res
            print(f"→ Using SMOTE-resampled data for {name}")
        else:
            X_used, y_used = X_train, y_train
            print(f"→ Using original training data for {name}")

        with mlflow.start_run(run_name=name):
            model.fit(X_used, y_used)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc = roc_auc_score(y_test, preds)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics({
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1,
                "ROC-AUC": roc
            })
            mlflow.sklearn.log_model(model, artifact_path=name.replace(" ", "_"))

            print(f"{name} → Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
