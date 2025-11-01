# create_corpus.py
# Build corpus.jsonl by merging APY production (state-year-crop) with IMD rainfall (state-year)
# Output lines like: "In 2010, KARNATAKA produced X tonnes of RICE across Y ha (yield Z t/ha). Annual rainfall: W mm."

import json
import pandas as pd
import numpy as np

APY_CSV = "APY.csv"
IMD_CSV = "Sub_Division_IMD_2017.csv"
OUT_PATH = "corpus.jsonl"

# Minimal subdivision→state map (extend if needed)
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

def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def load_apy(path):
    df = pd.read_csv(path)
    df = normalize_cols(df)
    rename = {
        "state_name":"state", "state":"state",
        "district_name":"district", "district":"district",
        "crop_year":"year", "year":"year",
        "crop":"crop", "season":"season",
        "area":"area_ha", "production":"production_ton"
    }
    for k,v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k:v})
    # Ensure cols
    for c in ["state","district","crop","year","area_ha","production_ton"]:
        if c not in df.columns:
            df[c] = None
    df["state"] = df["state"].astype(str).str.upper().str.strip()
    df["crop"]  = df["crop"].astype(str).str.upper().str.strip()
    df["year"]  = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce")
    df["production_ton"] = pd.to_numeric(df["production_ton"], errors="coerce")
    return df

def load_imd(path):
    rf = pd.read_csv(path)
    rf = normalize_cols(rf)
    rf.columns = [c.upper() for c in rf.columns]
    # prefer ANNUAL if present, else sum months
    if "ANNUAL" in rf.columns:
        rf["ANNUAL_MM"] = pd.to_numeric(rf["ANNUAL"], errors="coerce")
    else:
        months = [c for c in rf.columns if c in
                  ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]]
        if months:
            rf["ANNUAL_MM"] = rf[months].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        else:
            if "RAINFALL" in rf.columns:
                rf["ANNUAL_MM"] = pd.to_numeric(rf["RAINFALL"], errors="coerce")
            else:
                rf["ANNUAL_MM"] = np.nan

    if "SUBDIVISION" not in rf.columns:
        cand = [c for c in rf.columns if "SUBDIV" in c.upper()]
        if cand:
            rf = rf.rename(columns={cand[0]:"SUBDIVISION"})
    if "YEAR" not in rf.columns:
        cand = [c for c in rf.columns if c.upper() == "YR" or c.upper().endswith("YEAR")]
        if cand:
            rf = rf.rename(columns={cand[0]:"YEAR"})

    rf["YEAR"] = pd.to_numeric(rf["YEAR"], errors="coerce").astype("Int64")
    rf["SUBDIVISION"] = rf["SUBDIVISION"].astype(str).upper().str.strip()
    rf["STATE"] = rf["SUBDIVISION"].map(SUBDIV_TO_STATE).fillna(rf["SUBDIVISION"])
    state_year = rf.groupby(["STATE","YEAR"], as_index=False)["ANNUAL_MM"].mean()
    state_year["STATE"] = state_year["STATE"].astype(str).str.upper().str.strip()
    return state_year.rename(columns={"STATE":"state","YEAR":"year","ANNUAL_MM":"annual_mm"})

def main():
    apy = load_apy(APY_CSV)
    imd = load_imd(IMD_CSV)

    # State-year-crop aggregates
    syc = apy.groupby(["state","year","crop"], as_index=False).agg(
        production_ton=("production_ton","sum"),
        area_ha=("area_ha","sum")
    )
    syc["yield_t_ha"] = syc["production_ton"] / syc["area_ha"].replace(0, np.nan)

    # Merge rainfall
    merged = syc.merge(imd, on=["state","year"], how="left")

    # Emit facts
    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for _, r in merged.iterrows():
            state = str(r["state"]).upper()
            crop  = str(r["crop"]).upper()
            year  = int(r["year"]) if pd.notnull(r["year"]) else None
            prod  = float(r["production_ton"]) if pd.notnull(r["production_ton"]) else None
            area  = float(r["area_ha"]) if pd.notnull(r["area_ha"]) else None
            yld   = float(r["yield_t_ha"]) if pd.notnull(r["yield_t_ha"]) else None
            rain  = float(r["annual_mm"]) if pd.notnull(r["annual_mm"]) else None

            if year is None or state == "" or crop == "":
                continue

            parts = []
            parts.append("In %d, %s produced %s tonnes of %s" %
                        (year, state, f"{prod:,.2f}" if prod is not None else "NA", crop))
            if area is not None:
                parts.append("across %s ha" % f"{area:,.2f}")
            if yld is not None:
                parts.append("(yield %s t/ha)" % f"{yld:,.2f}")
            sent = " ".join(parts).strip()
            if rain is not None:
                sent += ". Annual rainfall: %s mm." % f"{rain:,.1f}"

            obj = {
                "text": sent,
                "meta": {"state": state, "crop": crop, "year": year}
            }
            out.write(json.dumps(obj) + "\n")
            n += 1

    print("Wrote %d facts to %s" % (n, OUT_PATH))

if __name__ == "__main__":
    main()
