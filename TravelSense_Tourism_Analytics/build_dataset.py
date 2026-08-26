"""Create a deterministic, realistic sample tourism dataset (CSV only)."""
from pathlib import Path
import csv
import math
import random

DESTINATIONS = {
    "Goa": ("Beach", 26000, 1.28),
    "Jaipur": ("Heritage", 21000, 1.20),
    "Manali": ("Hill Station", 18500, 1.30),
    "Kerala": ("Nature", 23500, 1.18),
    "Agra": ("Heritage", 24500, 1.15),
    "Darjeeling": ("Hill Station", 16000, 1.22),
}
MONTH_FACTORS = [0.62, 0.68, 0.86, 0.92, 0.98, 1.07, 0.83, 0.88, 0.96, 1.10, 1.32, 1.48]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def season_for_month(month):
    if month in (12, 1, 2): return "Winter"
    if month in (3, 4, 5): return "Summer"
    if month in (6, 7, 8, 9): return "Monsoon"
    return "Autumn"


def create_dataset(output_path):
    random.seed(42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Destination", "Category", "Year", "Month", "Month_Name", "Season", "Domestic_Tourists", "Foreign_Tourists", "Hotel_Occupancy", "Festival_Event", "Tourist_Footfall"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for destination, (category, base, foreign_share) in DESTINATIONS.items():
            for year in range(2019, 2025):
                growth = 1 + (year - 2019) * 0.055
                for month in range(1, 13):
                    festival = int(month in (10, 11, 12) or (destination == "Goa" and month == 2) or (destination == "Manali" and month == 6))
                    seasonal = MONTH_FACTORS[month - 1]
                    noise = random.uniform(0.93, 1.07)
                    total = int(base * growth * seasonal * (1.10 if festival else 1) * noise)
                    foreign = int(total * (0.075 * foreign_share + random.uniform(-0.012, 0.012)))
                    domestic = total - foreign
                    occupancy = round(min(96, max(42, 46 + seasonal * 30 + festival * 10 + random.uniform(-5, 5))), 1)
                    writer.writerow({"Destination": destination, "Category": category, "Year": year, "Month": month, "Month_Name": MONTH_NAMES[month - 1], "Season": season_for_month(month), "Domestic_Tourists": domestic, "Foreign_Tourists": foreign, "Hotel_Occupancy": occupancy, "Festival_Event": festival, "Tourist_Footfall": total})


if __name__ == "__main__":
    create_dataset(Path(__file__).resolve().parent / "dataset" / "tourism_data.csv")
    print("Dataset created successfully.")
