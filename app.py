import random

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb

MODEL_PATH = "xgb.h5"


def load_model():
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_PATH)
    return xgb_model


model = load_model()

feature_names = [
    "product_id",
    "units",
    "weight",
    "material_handling",
    "weight_class",
    "customer_Amsterdam",
    "customer_Athens",
    "customer_Barcelona",
    "customer_Berlin",
    "customer_Bordeaux",
    "customer_Bremen",
    "customer_Bucharest",
    "customer_Budapest",
    "customer_Cologne",
    "customer_Copenhagen",
    "customer_Hanover",
    "customer_Helsinki",
    "customer_Lisbon",
    "customer_Lyon",
    "customer_Madrid",
    "customer_Malmö",
    "customer_Marseille",
    "customer_Milan",
    "customer_Munich",
    "customer_Naples",
    "customer_Paris",
    "customer_Porto",
    "customer_Prague",
    "customer_Rome",
    "customer_Stockholm",
    "customer_Turin",
    "customer_Valencia",
    "customer_Vienna",
    "origin_port_Athens",
    "origin_port_Barcelona",
    "origin_port_Rotterdam",
    "3pl_v_001",
    "3pl_v_002",
    "3pl_v_003",
    "3pl_v_004",
    "customs_procedures_CRF",
    "customs_procedures_DTD",
    "customs_procedures_DTP",
    "logistic_hub_-1",
    "logistic_hub_Bratislava",
    "logistic_hub_Dusseldorf",
    "logistic_hub_Hamburg",
    "logistic_hub_Liege",
    "logistic_hub_Lille",
    "logistic_hub_Rome",
    "logistic_hub_Venlo",
    "logistic_hub_Warsaw",
    "logistic_hub_Zaragoza",
]

customer_names = [
    "customer_Amsterdam",
    "customer_Athens",
    "customer_Barcelona",
    "customer_Berlin",
    "customer_Bordeaux",
    "customer_Bremen",
    "customer_Bucharest",
    "customer_Budapest",
    "customer_Cologne",
    "customer_Copenhagen",
    "customer_Hanover",
    "customer_Helsinki",
    "customer_Lisbon",
    "customer_Lyon",
    "customer_Madrid",
    "customer_Malmö",
    "customer_Marseille",
    "customer_Milan",
    "customer_Munich",
    "customer_Naples",
    "customer_Paris",
    "customer_Porto",
    "customer_Prague",
    "customer_Rome",
    "customer_Stockholm",
    "customer_Turin",
    "customer_Valencia",
    "customer_Vienna",
]

origin_port_names = [
    "origin_port_Athens",
    "origin_port_Barcelona",
    "origin_port_Rotterdam",
]

logistic_hub_names = [
    "logistic_hub_-1",
    "logistic_hub_Bratislava",
    "logistic_hub_Dusseldorf",
    "logistic_hub_Hamburg",
    "logistic_hub_Liege",
    "logistic_hub_Lille",
    "logistic_hub_Rome",
    "logistic_hub_Venlo",
    "logistic_hub_Warsaw",
    "logistic_hub_Zaragoza",
]

customs_procedures_names = [
    "customs_procedures_CRF",
    "customs_procedures_DTD",
    "customs_procedures_DTP",
]


# Gradio predict function
def predict(*args):
    # Convert categorical inputs to one-hot encoding from feature_names

    df = pd.DataFrame(columns=feature_names)
    origin_port = args[0]
    pl = args[1]
    customs_procedures = args[2]
    logistic_hub = args[3]
    customer = args[4]
    product_id = args[5]
    units = args[6]
    weight = args[7]
    material_handling = args[8]
    weight_class = args[9]
    print(args)

    # Get prediction
    pred = model.predict(df)[0]
    # Get probability
    prob = model.predict_proba(df)[0][pred]

    # Return prediction and probability
    return pred, prob


def interpret(*args):
    ...


#     # Convert args to dataframe
#     df = pd.DataFrame([args], columns=feature_names)
#     # Get prediction
#     pred = model.predict(df)[0]
#     # Get probability
#     prob = model.predict_proba(df)[0][pred]

#     # Get SHAP values
#     explainer = shap.TreeExplainer(model)

#     # Return prediction and probability return pred, prob


df = pd.read_csv("data/dataframefinal.csv", sep=",")

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                origin_port = gr.Dropdown(
                    label="Origin Port",
                    choices=list(df.origin_port.unique()),
                )
                pl = gr.Dropdown(
                    label="Third-party logistics company",
                    choices=list(df["3pl"].unique()),
                )
                customs_procedures = gr.Dropdown(
                    label="Customs Procedures",
                    choices=list(df.customs_procedures.unique()),
                )
                logistic_hub = gr.Dropdown(
                    label="Logistic Hub",
                    choices=list(df.logistic_hub.unique()),
                )
            with gr.Column():
                customer = gr.Dropdown(
                    label="Customer",
                    choices=list(df.customer.unique()),
                )
                product_id = gr.Dropdown(
                    label="Product ID",
                    choices=list(str(df.product_id.unique())),
                )
                units = gr.Slider(
                    label="Units",
                    max=1000,
                )
                weight = gr.Slider(
                    label="Weight",
                    max=1000,
                )
                material_handling = gr.Slider(
                    label="Material Handling",
                    max=5,
                )
                weight_class = gr.Slider(
                    label="Weight Class",
                    max=5,
                )

        with gr.Row():
            with gr.Column():
                predict_btn = gr.Button(value="Predict")
                interpret_btn = gr.Button(value="Explain")
            label = gr.Label()
            plot = gr.Plot()
            predict_btn.click(
                predict,
                inputs=[
                    origin_port,
                    pl,
                    customs_procedures,
                    logistic_hub,
                    customer,
                    product_id,
                    units,
                    weight,
                    material_handling,
                    weight_class,
                ],
                outputs=[label],
            )
            interpret_btn.click(
                interpret,
                inputs=[
                    origin_port,
                    pl,
                    customs_procedures,
                    logistic_hub,
                    customer,
                    product_id,
                    units,
                    weight,
                    material_handling,
                    weight_class,
                ],
                outputs=[plot],
            )

demo.launch()
