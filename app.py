from src.error import CustomException
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
from pipeline.predict_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)


# Cretaing the routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/getpara", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("writing_score")),
            writing_score=float(request.form.get("reading_score")),
        )
        pred_df = data.get_data_as_data_frame()
        print(data)
        predict_pipleine = PredictionPipeline()
        result = predict_pipleine.predict(pred_df)
        print(result)
        return render_template("home.html", results=result[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
