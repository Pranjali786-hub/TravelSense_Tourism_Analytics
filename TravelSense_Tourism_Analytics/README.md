# TravelSense Tourism Analytics Platform

TravelSense is a beginner-friendly Flask internship project that analyses tourism data and predicts tourist footfall with TensorFlow. It deliberately uses **CSV files only**—there is no database and no external API.

## Features

- Static login: `admin` / `Admin@123`
- Responsive Bootstrap dashboard with CSV-based statistics
- Destination, month, season, year, popularity, and visitor-type analysis
- Matplotlib and Seaborn charts rendered on the dashboard
- TensorFlow neural-network prediction form
- Separate prediction report with model inputs and a simple explanation
- Pickle files for the feature scaler and category encoders

## Technology used

Python, Flask, Pandas, NumPy, Matplotlib, Seaborn, TensorFlow, Pickle, HTML, CSS, JavaScript and Bootstrap.

## Folder structure

```text
TravelSense_Tourism_Analytics/
├── app.py                 # Flask routes and dashboard logic
├── build_dataset.py       # Creates the included realistic CSV dataset
├── train_model.py         # TensorFlow training and artifact saving
├── requirements.txt
├── dataset/
│   └── tourism_data.csv   # Created automatically if it is absent
├── model/
│   ├── tourism_model.keras
│   ├── scaler.pkl
│   └── encoders.pkl
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   ├── prediction.html
│   └── report.html
└── static/
    ├── css/style.css
    ├── js/script.js
    └── images/
```

## Dataset

The deterministic sample dataset contains monthly records from 2019–2024 for Goa, Jaipur, Manali, Kerala, Agra and Darjeeling. Its columns are:

`Destination`, `Category`, `Year`, `Month`, `Month_Name`, `Season`, `Domestic_Tourists`, `Foreign_Tourists`, `Hotel_Occupancy`, `Festival_Event`, and `Tourist_Footfall`.

`build_dataset.py` produces the CSV from a fixed random seed, so every user receives the same realistic sample records. The Flask app runs this automatically if the CSV is absent.

## Installation and running

Use Python 3.10 or 3.11. In the VS Code terminal, open this project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

If your computer uses another supported Python command, replace `py -3.11` with `python`.

On first start, the app automatically creates `dataset/tourism_data.csv`, trains the TensorFlow model, and saves:

- `model/tourism_model.keras`
- `model/scaler.pkl`
- `model/encoders.pkl`

This can take a short time only on the first run. Then open the local address printed in the terminal (normally `http://127.0.0.1:5000`).

## Optional manual data generation and training

```powershell
python build_dataset.py
python train_model.py
python app.py
```

Run these if you change the dataset and want a freshly trained model.

## Login credentials

| Username | Password |
|---|---|
| `admin` | `Admin@123` |

## How to use

1. Log in and inspect the dashboard cards and charts.
2. Select **Predict Footfall**.
3. Enter a destination, season, year, month, expected domestic and foreign visitors, hotel occupancy, and event status.
4. Select **Predict Tourist Footfall**.
5. Open **View detailed report** to see the input values, prediction, historical destination average, and simple interpretation.

## Explanation for a viva

The model first converts Destination and Season from text into numbers using label dictionaries. It then standardizes all eight input features using their mean and standard deviation. A small TensorFlow neural network (32 neurons, then 16 neurons, then one output) learns the relationship between the input conditions and `Tourist_Footfall`. The model, scaler, and encoders are saved so the Flask website can make future predictions with the same preprocessing.

## Important notes

- This is an educational sample model trained on generated realistic-looking CSV data; its predictions are demonstrations, not official forecasts.
- No username/password database is used. The static login is intentionally simple for the internship demonstration.
