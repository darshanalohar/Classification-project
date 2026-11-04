from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)



model = joblib.load("model.pkl")


@app.route("/")
def home():

    return render_template(
        "index.html"
    )



@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:


        sepal_length = float(
            request.form["sepal_length"]
        )

        sepal_width = float(
            request.form["sepal_width"]
        )

        petal_length = float(
            request.form["petal_length"]
        )

        petal_width = float(
            request.form["petal_width"]
        )


        input_data = pd.DataFrame({

            "sepal_length": [
                sepal_length
            ],

            "sepal_width": [
                sepal_width
            ],

            "petal_length": [
                petal_length
            ],

            "petal_width": [
                petal_width
            ]

        })



        prediction = model.predict(
            input_data
        )[0]


        return render_template(
            "index.html",
            prediction=prediction
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )