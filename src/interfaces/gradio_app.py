import gradio as gr
from model import load_data, build_model, train_model, predict
import pandas as pd

# Load and preprocess data
training_df = load_data()

# Build and train the model
learning_rate = 0.07
epochs = 20
batch_size = 35
my_feature = "median_income"
my_label = "median_house_value"
model = build_model(learning_rate)
train_model(model, training_df, my_feature, my_label, epochs, batch_size)

# Define the Gradio interface
def gradio_predict(input_data):
    input_df = pd.DataFrame({my_feature: [float(input_data)]})
    prediction = predict(model, input_df)
    return prediction[0][0]

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.inputs.Textbox(lines=1, placeholder="Enter median income..."),
    outputs="text",
    title="House Value Prediction",
    description="Predict the median house value based on median income"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
