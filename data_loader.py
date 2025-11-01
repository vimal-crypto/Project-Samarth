
import numpy as np
import pandas as pd

YEAR_MIN, YEAR_MAX = 1997, 2017

# Minimal IMD subdivision → state mapping (extend if you have time)
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

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def load_apy(path_or_buf) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf)
    df = _normalize_cols(df)
    df = df.rename(columns={
        "state_name": "state",
        "district_name": "district",
        "crop_year": "year",
        "area": "area_ha",
        "production": "production_ton"
    })
    # required cols
    for c in ["state","district","crop","year","area_ha","production_ton","season"]:
        if c not in df.columns:
            df[c] = np.nan

    # normalize types
    for c in ["state","district","crop","season"]:
        df[c] = df[c].astype(str).str.upper().str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce")
    df["production_ton"] = pd.to_numeric(df["production_ton"], errors="coerce")

    # filter to 1997–2017 & remove obvious noise
    df = df[(df["year"]>=YEAR_MIN) & (df["year"]<=YEAR_MAX)]
    df = df[(df["production_ton"].fillna(0) >= 1.0) & (df["area_ha"].fillna(0) >= 0.5)]
    return df

def load_imd(path_or_buf) -> pd.DataFrame:
    rf = pd.read_csv(path_or_buf)
    rf.columns = [c.upper().strip() for c in rf.columns]

    # unify columns
    if "YEAR" not in rf.columns:
        guess = [c for c in rf.columns if c.endswith("YEAR") or c in ("YR","YEAR_")]
        if guess: rf = rf.rename(columns={guess[0]:"YEAR"})
    if "SUBDIVISION" not in rf.columns:
        guess = [c for c in rf.columns if "SUBDIV" in c]
        if guess: rf = rf.rename(columns={guess[0]:"SUBDIVISION"})

    # monthly → annual
    mcols = [c for c in rf.columns if c in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
    if mcols:
        rf["ANNUAL_MM"] = rf[mcols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    elif "ANNUAL" in rf.columns:
        rf["ANNUAL_MM"] = pd.to_numeric(rf["ANNUAL"], errors="coerce")
    else:
        rf["ANNUAL_MM"] = np.nan

    rf["YEAR"] = pd.to_numeric(rf["YEAR"], errors="coerce").astype("Int64")
    rf["SUBDIVISION"] = rf["SUBDIVISION"].astype(str).str.upper().str.strip()
    rf["STATE"] = rf["SUBDIVISION"].map(SUBDIV_TO_STATE).fillna(rf["SUBDIVISION"])

    # aggregate to state-year
    state_year = rf.groupby(["STATE","YEAR"], as_index=False)["ANNUAL_MM"].mean()
    state_year = state_year[(state_year["YEAR"]>=YEAR_MIN) & (state_year["YEAR"]<=YEAR_MAX)]
    state_year["STATE"] = state_year["STATE"].astype(str).str.upper().str.strip()

    return state_year.rename(columns={"STATE":"state","YEAR":"year","ANNUAL_MM":"annual_mm"})
