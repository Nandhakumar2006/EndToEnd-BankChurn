import gradio as gr
import pickle
import pandas as pd

with open("all_models.pkl", "rb") as f:
    models, features = pickle.load(f)

default_model = "Random Forest"

def predict_selected(model_name, CreditScore, Age, Tenure, Balance,
                     NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
                     Geography, Gender):

    input_dict = {
        "CreditScore": float(CreditScore),
        "Age": float(Age),
        "Tenure": float(Tenure),
        "Balance": float(Balance),
        "NumOfProducts": float(NumOfProducts),
        "HasCrCard": float(HasCrCard),
        "IsActiveMember": float(IsActiveMember),
        "EstimatedSalary": float(EstimatedSalary),
        "Geography_Germany": 1 if Geography == "Germany" else 0,
        "Geography_Spain": 1 if Geography == "Spain" else 0,
        "Gender_Male": 1 if Gender == "Male" else 0
    }

    X_input = pd.DataFrame([input_dict])
    model = models[model_name]

    try:
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None:
            X_input = X_input.reindex(columns=model_features, fill_value=0)
        else:
            expected_features = getattr(model, "n_features_in_", len(features))
            if X_input.shape[1] > expected_features:
                X_input = X_input.iloc[:, :expected_features]
            elif X_input.shape[1] < expected_features:
                for i in range(expected_features - X_input.shape[1]):
                    X_input[f"missing_{i}"] = 0
    except Exception:
        X_input = X_input.reindex(columns=features, fill_value=0)

    try:
        pred_prob = model.predict_proba(X_input)[0][1]
    except Exception:
        pred_prob = None
    pred = model.predict(X_input)[0]

    result_text = f" Prediction: {'Customer Exits' if pred == 1 else 'Customer Stays'}"
    if pred_prob is not None:
        result_text += f"\n Exit Probability: {pred_prob * 100:.2f}%"
    return result_text


with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber")) as app:  # Only primary_hue
    gr.Markdown(
        """
        <h1 style='text-align:left; color:#d19a66;'>🏦 Customer Churn Prediction</h1>
        <p style='text-align:left; color:#e6e6e6;'>
        Choose a model (Random Forest recommended), enter customer details, 
        and predict whether the customer will exit.
        </p>
        """,
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(models.keys()),
                value=default_model,
                label="Model (Recommended: Random Forest)"
            )

            CreditScore = gr.Number(label="Credit Score", value=650)
            Age = gr.Number(label="Age", value=35)
            Tenure = gr.Number(label="Tenure (Years)", value=5)
            Balance = gr.Number(label="Balance", value=50000)
            NumOfProducts = gr.Number(label="Number of Products", value=1)
            EstimatedSalary = gr.Number(label="Estimated Salary", value=60000)
            HasCrCard = gr.Radio(choices=["0", "1"], label="Has Credit Card (1 = Yes, 0 = No)", value="1")
            IsActiveMember = gr.Radio(choices=["0", "1"], label="Is Active Member (1 = Yes, 0 = No)", value="1")
            Geography = gr.Dropdown(choices=["France", "Germany", "Spain"], label="Geography", value="France")
            Gender = gr.Dropdown(choices=["Male", "Female"], label="Gender", value="Male")

            predict_btn = gr.Button("🔍 Predict", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="Prediction Result",
                lines=5,
                interactive=False,
                elem_id="result_box"
            )

    gr.Markdown(
        """
        <p style='text-align:left; color:#bfbfbf; font-size:0.9em;'>
        Developed by <b>AI Workforce</b> — Elegant Left Layout 🌗
        </p>
        """,
    )

    app.load(
        None,
        js="""
        () => {
            const box = document.querySelector('#result_box');
            if (box) {
                box.style.fontSize = '1.2em';
                box.style.borderRadius = '12px';
                box.style.background = '#2a2a2a';
                box.style.color = '#ffd28a';
                box.style.height = '180px';
            }
        }
        """,
    )

    predict_btn.click(
        predict_selected,
        inputs=[
            model_dropdown, CreditScore, Age, Tenure, Balance,
            NumOfProducts, HasCrCard, IsActiveMember,
            EstimatedSalary, Geography, Gender
        ],
        outputs=output_box
    )

if __name__ == "__main__":
    app.launch()

