from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("logreg_rfe_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature names as used in training
feature_names = ['const', 'GRE Score', 'University Rating', 'CGPA']

def prepare_input(gre, rating, cgpa):
    # Include intercept term
    data = [[1.0, gre, rating, cgpa]]
    return pd.DataFrame(data, columns=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    # try:
    gre = float(request.form["gre"])
    rating = float(request.form["rating"])
    cgpa = float(request.form["cgpa"])
    input_df = prepare_input(gre, rating, cgpa)
    prob = model.predict(input_df)[0]
    label = "Admit" if prob >= 0.6 else "Reject"
    print(label)
    return jsonify({"label": label, "probability": round(float(prob), 4)})
    # except Exception as e:
    #     return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
