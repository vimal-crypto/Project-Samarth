import pandas as pd
import numpy as np
import json
import os

# -------------------------
# File paths (change if needed)
# -------------------------
CROP_FILE = "APY.csv"
RAIN_FILE = "Sub_Division_IMD_2017.csv"
OUTPUT_FILE = "corpus.jsonl"

# -------------------------
# IMD Subdivision → State Mapping
# -------------------------
SUBDIV_TO_STATE = {
    "TAMIL NADU": "TAMIL NADU",
    "PUDUCHERRY AND KARAIKAL": "TAMIL NADU",
    "KERALA": "KERALA",
    "COASTAL KARNATAKA": "KARNATAKA",
    "SOUTH INTERIOR KARNATAKA": "KARNATAKA",
    "NORTH INTERIOR KARNATAKA": "KARNATAKA",
    "MADHYA MAHARASHTRA": "MAHARASHTRA",
    "MARATHWADA": "MAHARASHTRA",
    "KONKAN & GOA": "MAHARASHTRA",
    "VIDARBHA": "MAHARASHTRA",
    "COASTAL ANDHRA PRADESH": "ANDHRA PRADESH",
    "NORTH INTERIOR ANDHRA PRADESH": "ANDHRA PRADESH",
    "TELANGANA": "TELANGANA",
    "RAYALASEEMA": "ANDHRA PRADESH",
    "EAST RAJASTHAN": "RAJASTHAN",
    "WEST RAJASTHAN": "RAJASTHAN",
    "EAST MADHYA PRADESH": "MADHYA PRADESH",
    "WEST MADHYA PRADESH": "MADHYA PRADESH",
    "EAST UTTAR PRADESH": "UTTAR PRADESH",
    "WEST UTTAR PRADESH": "UTTAR PRADESH",
    "BIHAR": "BIHAR",
    "JHARKHAND": "JHARKHAND",
    "ODISHA": "ODISHA",
    "GANGETIC WEST BENGAL": "WEST BENGAL",
    "SUB-HIMALAYAN WEST BENGAL & SIKKIM": "WEST BENGAL",
    "ASSAM & MEGHALAYA": "ASSAM",
    "NAGA MANI MIZO & TRIPURA": "TRIPURA",
    "ARUNACHAL PRADESH": "ARUNACHAL PRADESH",
    "UTTARAKHAND": "UTTARAKHAND",
    "HIMACHAL PRADESH": "HIMACHAL PRADESH",
    "JAMMU & KASHMIR": "JAMMU & KASHMIR",
    "SAURASHTRA & KUTCH": "GUJARAT",
    "GUJARAT REGION": "GUJARAT",
    "GOA": "GOA"
}


# -------------------------
# Load and normalize APY dataset
# -------------------------
def load_crop_data():
    df = pd.read_csv(CROP_FILE)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={
        "state_name": "state",
        "district_name": "district",
        "crop_year": "year",
        "area": "area_ha",
        "production": "production_ton"
    })

    df["state"] = df["state"].astype(str).str.upper().str.strip()
    df["crop"] = df["crop"].astype(str).str.upper().str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce")
    df["production_ton"] = pd.to_numeric(df["production_ton"], errors="coerce")

    # filter valid rows
    df = df[(df["year"] >= 1997) & (df["year"] <= 2017)]
    df = df[df["production_ton"] >= 1.0]  # remove noise
    df = df[df["area_ha"] >= 0.5]        # min usable area
    return df


# -------------------------
# Load and normalize rainfall dataset
# -------------------------
def load_rainfall_data():
    df = pd.read_csv(RAIN_FILE)
    df.columns = [c.upper().strip() for c in df.columns]
    df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.upper().str.strip()
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

    month_cols = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    df["ANNUAL_MM"] = df[month_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    df["STATE"] = df["SUBDIVISION"].map(SUBDIV_TO_STATE).fillna(df["SUBDIVISION"])
    df = df.groupby(["STATE", "YEAR"], as_index=False)["ANNUAL_MM"].mean()

    df = df[(df["YEAR"] >= 1997) & (df["YEAR"] <= 2017)]
    return df


# -------------------------
# Generate corpus JSONL
# -------------------------
def generate_corpus():
    crop = load_crop_data()
    rain = load_rainfall_data()

    # state-year aggregation
    crop_state_year = crop.groupby(["state", "crop", "year"], as_index=False).agg(
        {"production_ton": "sum", "area_ha": "sum"}
    )
    crop_state_year["yield_ton_per_ha"] = crop_state_year["production_ton"] / crop_state_year["area_ha"]

    # join rainfall
    merged = crop_state_year.merge(rain, left_on=["state", "year"], right_on=["STATE", "YEAR"], how="left")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            text = (
                f"In {row['year']}, {row['state']} produced {row['production_ton']:.2f} tonnes of "
                f"{row['crop']} across {row['area_ha']:.2f} hectares "
                f"(yield: {row['yield_ton_per_ha']:.2f} t/ha). "
                f"The state received {row['ANNUAL_MM']:.1f} mm of rainfall that year. "
                "(Sources: APY, IMD)"
            )

            obj = {
                "text": text,
                "meta": {
                    "state": row["state"],
                    "crop": row["crop"],
                    "year": int(row["year"]),
                }
            }
            f.write(json.dumps(obj) + "\n")

    print(f"✅ Corpus saved to {OUTPUT_FILE} — {len(merged)} records written.")


if __name__ == "__main__":
    generate_corpus()
