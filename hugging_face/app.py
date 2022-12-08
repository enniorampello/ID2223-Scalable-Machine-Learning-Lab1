import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login(api_key_value="API_KEY")
fs = project.get_feature_store()
#q

mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

def titanic(pclass, sex, age, sibsp, parch, fare, embarked):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(fare)
    input_list.append(embarked)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    ret_str = "Survived" if res[0] == 1 else "Not survived"
    return ret_str

        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Predictive Analytics",
    description="Experiment to predict if a passenger survived the Titanic disaster",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="PClass"),
        gr.inputs.Number(default=1.0, label="Sex: Female = 0, Male = 1"),
        gr.inputs.Number(default=1.0, label="Age"),
        gr.inputs.Number(default=1.0, label="SibSp"),
        gr.inputs.Number(default=1.0, label="Parch"),
        gr.inputs.Number(default=1.0, label="Fare"),
        gr.inputs.Number(default=1.0, label="Embarked: S = 0, C = 1, Q = 2"),
        ],
    outputs=gr.Textbox())

demo.launch()

