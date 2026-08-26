from pathlib import Path
import base64
from io import BytesIO
import pickle
import subprocess
import sys
from functools import wraps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, session, url_for

from train_model import FEATURE_COLUMNS, prepare_features

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "tourism_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "tourism_model.keras"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
ENCODERS_FILE = MODEL_DIR / "encoders.pkl"

app = Flask(__name__)
app.config["SECRET_KEY"] = "travelsense-internship-demo-key"


def ensure_project_files():
    """Create the included data and model artifacts on first run, if needed."""
    if not DATA_FILE.exists():
        subprocess.run([sys.executable, str(BASE_DIR / "build_dataset.py")], check=True)
    if not all(path.exists() for path in (MODEL_FILE, SCALER_FILE, ENCODERS_FILE)):
        subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], check=True)


ensure_project_files()
DATA = pd.read_csv(DATA_FILE).dropna().drop_duplicates()
MODEL = tf.keras.models.load_model(MODEL_FILE)
with SCALER_FILE.open("rb") as file: SCALER = pickle.load(file)
with ENCODERS_FILE.open("rb") as file: ENCODERS = pickle.load(file)


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


def chart_to_base64(plot_function):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_function(ax)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def dashboard_charts():
    by_destination = DATA.groupby("Destination")["Tourist_Footfall"].sum().sort_values(ascending=False)
    by_month = DATA.groupby("Month")["Tourist_Footfall"].mean()
    by_season = DATA.groupby("Season")["Tourist_Footfall"].mean().reindex(["Winter", "Summer", "Monsoon", "Autumn"])
    by_year = DATA.groupby("Year")["Tourist_Footfall"].sum()
    visitors = DATA[["Domestic_Tourists", "Foreign_Tourists"]].sum()
    return {
        "destination": chart_to_base64(lambda ax: by_destination.plot.bar(ax=ax, color="#0e7490") or ax.set_ylabel("Total tourists")),
        "monthly": chart_to_base64(lambda ax: ax.plot(by_month.index, by_month.values, marker="o", color="#ea580c") or ax.set(xlabel="Month", ylabel="Average tourists")),
        "seasonal": chart_to_base64(lambda ax: by_season.plot.bar(ax=ax, color="#7c3aed") or ax.set_ylabel("Average tourists")),
        "yearly": chart_to_base64(lambda ax: by_year.plot(ax=ax, marker="o", color="#16a34a") or ax.set_ylabel("Total tourists")),
        "visitor_type": chart_to_base64(lambda ax: visitors.plot.pie(ax=ax, autopct="%1.1f%%", colors=["#0284c7", "#f59e0b"]) or ax.set_ylabel("")),
    }


@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        if request.form.get("username") == "admin" and request.form.get("password") == "Admin@123":
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    destination_totals = DATA.groupby("Destination")["Tourist_Footfall"].sum()
    stats = {"total": int(DATA["Tourist_Footfall"].sum()), "destinations": DATA["Destination"].nunique(), "popular": destination_totals.idxmax(), "average": int(DATA["Tourist_Footfall"].mean()), "highest": int(DATA["Tourist_Footfall"].max())}
    return render_template("dashboard.html", stats=stats, charts=dashboard_charts(), recent=DATA.sort_values(["Year", "Month"], ascending=False).head(8).to_dict("records"))


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    destinations = sorted(ENCODERS["Destination"])
    seasons = ["Winter", "Summer", "Monsoon", "Autumn"]
    if request.method == "POST":
        try:
            input_data = {"Destination": request.form["destination"], "Season": request.form["season"], "Year": int(request.form["year"]), "Month": int(request.form["month"]), "Domestic_Tourists": int(request.form["domestic"]), "Foreign_Tourists": int(request.form["foreign"]), "Hotel_Occupancy": float(request.form["occupancy"]), "Festival_Event": int(request.form["festival"])}
            if not 1 <= input_data["Month"] <= 12 or min(input_data["Domestic_Tourists"], input_data["Foreign_Tourists"]) < 0: raise ValueError
            input_frame = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
            features, _, _ = prepare_features(input_frame, ENCODERS, SCALER)
            predicted = max(0, int(round(float(MODEL.predict(features, verbose=0)[0][0]))))
            session["last_prediction"] = {**input_data, "Predicted_Tourist_Footfall": predicted}
            return render_template("prediction.html", destinations=destinations, seasons=seasons, prediction=predicted, form=input_data)
        except (KeyError, TypeError, ValueError):
            flash("Please enter valid non-negative numbers and select all fields.", "danger")
    return render_template("prediction.html", destinations=destinations, seasons=seasons, form={})


@app.route("/report")
@login_required
def report():
    prediction_data = session.get("last_prediction")
    if not prediction_data:
        flash("Make a prediction first to view its report.", "info")
        return redirect(url_for("prediction"))
    destination_average = int(DATA.loc[DATA["Destination"] == prediction_data["Destination"], "Tourist_Footfall"].mean())
    difference = prediction_data["Predicted_Tourist_Footfall"] - destination_average
    interpretation = "above" if difference >= 0 else "below"
    return render_template("report.html", item=prediction_data, destination_average=destination_average, difference=abs(difference), interpretation=interpretation)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
