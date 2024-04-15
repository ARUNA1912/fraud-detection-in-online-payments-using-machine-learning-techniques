import joblib
import gradio as gr
from pydantic import BaseModel

# 1. Load the trained model
model = joblib.load('frauddetection.pkl')

# 2. Define the input data schema using Pydantic BaseModel
class InputData(BaseModel):
    Year: int
    Month: int
    UseChip: int
    Amount: int
    MerchantName: int
    MerchantCity: int
    MerchantCountry: int
    mcc: int
    # Add the rest of the input features (feature4, feature5, ..., feature12)

# 3. Define the prediction function
def predict(year, month, use_chip, amount, merchant_name, merchant_city, merchant_country, mcc):
    # Perform the prediction using the loaded model
    prediction = model.predict([[year, month, use_chip, amount, merchant_name, merchant_city, merchant_country, mcc]])[0]  # Replace ... with the rest of the features

    # Convert the prediction to a string (or any other format you prefer)
    result = "Fraud" if prediction == 1 else "Not a Fraud"

    return result

# 4. Create a Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Year"),
        gr.Number(label="Month"),
        gr.Number(label="UseChip"),
        gr.Number(label="Amount"),
        gr.Number(label="MerchantName"),
        gr.Number(label="MerchantCity"),
        gr.Number(label="MerchantCountry"),
        gr.Number(label="mcc"),
        # Add the rest of the input features as individual Gradio input components
    ],
    outputs=gr.Textbox(),
)

# 5. Launch the Gradio interface
iface.launch()