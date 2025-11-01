# app.py — Project Samarth (Fast MVP: Analytics-only; no vector DB)
# Answers Q1–Q4 from APY.csv (production) + Sub_Division_IMD_2017.csv (rainfall)
# with DuckDB+pandas, charts, and explicit citations.

import os
import io
import re
import numpy as np
import pandas as pd
import duckdb
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Project Samarth — Agri × Climate (Fast MVP)", layout="wide")

# ---------- CONFIG ----------
DATA_DIR = "."
APY_FILE = "APY.csv"
IMD_FILE = "Sub_Division_IMD_2017.csv"
YEAR_MIN, YEAR_MAX = 1997, 2017

# IMD subdivision → state (minimal, extend as needed)
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

# ---------- HELPERS ----------
def _normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_apy(path_or_buf):
    df = pd.read_csv(path_or_buf)
    df = _normalize_cols(df)
    df = df.rename(columns={
        "state_name":"state", "district_name":"district",
        "crop_year":"year", "area":"area_ha", "production":"production_ton"
    })
    # keep required columns
    for c in ["state","district","crop","year","area_ha","production_ton","season"]:
        if c not in df.columns:
            df[c] = np.nan
    # normalize
    df["state"] = df["state"].astype(str).str.upper().str.strip()
    df["district"] = df["district"].astype(str).str.upper().str.strip()
    df["crop"] = df["crop"].astype(str).str.upper().str.strip()
    df["season"] = df["season"].astype(str).str.upper().str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce")
    df["production_ton"] = pd.to_numeric(df["production_ton"], errors="coerce")
    # Filter years + remove tiny noise
    df = df[(df["year"]>=YEAR_MIN) & (df["year"]<=YEAR_MAX)]
    df = df[(df["production_ton"].fillna(0) >= 1.0) & (df["area_ha"].fillna(0) >= 0.5)]
    return df

@st.cache_data(show_spinner=False)
def load_imd(path_or_buf):
    rf = pd.read_csv(path_or_buf)
    rf.columns = [c.upper().strip() for c in rf.columns]
    # unify required columns
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

def build_db(crop_df, rain_df):
    con = duckdb.connect(database=":memory:")
    con.register("crop_df", crop_df)
    con.register("rain_df", rain_df)

    con.execute("DROP TABLE IF EXISTS fact_crop")
    con.execute("""
        CREATE TABLE fact_crop AS
        SELECT
          UPPER(TRIM(state)) AS state,
          UPPER(TRIM(district)) AS district,
          UPPER(TRIM(crop)) AS crop,
          COALESCE(UPPER(TRIM(season)),'WHOLE YEAR') AS season,
          CAST(year AS INTEGER) AS year,
          CAST(area_ha AS DOUBLE) AS area_ha,
          CAST(production_ton AS DOUBLE) AS production_ton
        FROM crop_df
        WHERE year IS NOT NULL
    """)

    con.execute("DROP TABLE IF EXISTS fact_rain")
    con.execute("""
        CREATE TABLE fact_rain AS
        SELECT
          UPPER(TRIM(state)) AS state,
          CAST(year AS INTEGER) AS year,
          CAST(annual_mm AS DOUBLE) AS annual_mm
        FROM rain_df
        WHERE year IS NOT NULL
    """)

    # convenience views
    con.execute("""
        CREATE OR REPLACE VIEW v_state_crop_year AS
        SELECT state,crop,year,
               SUM(production_ton) AS prod_ton,
               SUM(area_ha) AS area_ha
        FROM fact_crop
        GROUP BY state,crop,year
    """)
    con.execute("""
        CREATE OR REPLACE VIEW v_state_year AS
        SELECT state,year,
               SUM(production_ton) AS prod_ton,
               SUM(area_ha) AS area_ha
        FROM fact_crop
        GROUP BY state,year
    """)
    return con

# ---------- ANALYTICS (Q1–Q4) ----------
def q1_rainfall_and_top_crops(con, state_x, state_y, crop_filter, n_years, top_m):
    # common last N years present in crop data for both states
    years_x = [r[0] for r in con.sql("SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC",[state_x]).fetchall()]
    years_y = [r[0] for r in con.sql("SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC",[state_y]).fetchall()]
    years = [y for y in years_x if y in years_y][:n_years]
    if not years:
        return {"error":"No overlapping years for those states in the dataset."}

    rain = con.sql(f"""
        SELECT state, year, annual_mm
        FROM fact_rain
        WHERE state IN (?,?) AND year IN ({",".join(["?"]*len(years))})
    """,[state_x,state_y,*years]).df()
    avg_rain = rain.groupby("state")["annual_mm"].mean().round(1).to_dict()

    crops = con.sql(f"""
        SELECT state, crop, SUM(prod_ton) AS prod
        FROM v_state_crop_year
        WHERE state IN (?,?) AND year IN ({",".join(["?"]*len(years))})
          AND crop LIKE ?
        GROUP BY state, crop
        ORDER BY state, prod DESC
    """,[state_x,state_y,*years,f"%{crop_filter}%"]).df()

    topx = crops[crops["state"]==state_x].nlargest(top_m,"prod")
    topy = crops[crops["state"]==state_y].nlargest(top_m,"prod")
    return {
        "years": years,
        "avg_rainfall_mm": avg_rain,
        "top_crops": {
            state_x: topx[["crop","prod"]].values.tolist(),
            state_y: topy[["crop","prod"]].values.tolist()
        }
    }

def q2_highest_lowest_district(con, crop_z, state_x, state_y):
    yx = con.sql("SELECT MAX(year) FROM fact_crop WHERE state=?",[state_x]).fetchone()[0]
    yy = con.sql("SELECT MAX(year) FROM fact_crop WHERE state=?",[state_y]).fetchone()[0]
    if not yx or not yy: return {"error":"No recent year for one/both states."}
    year = min(yx,yy)
    df = con.sql("""
        SELECT state,district,SUM(production_ton) AS prod
        FROM fact_crop
        WHERE year=? AND crop LIKE ?
        GROUP BY state,district
    """,[year,f"%{crop_z}%"]).df()
    hi_x = df[df["state"]==state_x].nlargest(1,"prod")
    lo_y = df[df["state"]==state_y].nsmallest(1,"prod")
    return {"year":year,
            "highest_in_state_x": hi_x.values.tolist(),
            "lowest_in_state_y": lo_y.values.tolist()}

def q3_trend_and_correlation(con, state, crop_c, horizon=10):
    years = [r[0] for r in con.sql(
        "SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC",[state]).fetchall()][:horizon]
    years_sorted = sorted(years)
    if not years_sorted: return {"error":"No years for region."}
    prod = con.sql(f"""
        SELECT year, SUM(prod_ton) AS prod
        FROM v_state_crop_year
        WHERE state=? AND crop LIKE ? AND year IN ({",".join(["?"]*len(years_sorted))})
        GROUP BY year ORDER BY year
    """,[state,f"%{crop_c}%",*years_sorted]).df()
    rain = con.sql(f"""
        SELECT year, AVG(annual_mm) AS rain
        FROM fact_rain
        WHERE state=? AND year IN ({",".join(["?"]*len(years_sorted))})
        GROUP BY year ORDER BY year
    """,[state,*years_sorted]).df()
    m = prod.merge(rain, on="year", how="inner")
    if len(m)>=3:
        r = float(np.corrcoef(m["prod"], m["rain"])[0,1])
    else:
        r = np.nan
    trend = "increasing" if m["prod"].diff().mean()>0 else "decreasing"
    return {"years":m["year"].tolist(),
            "production":m["prod"].round(2).tolist(),
            "rainfall_mm":m["rain"].round(1).tolist(),
            "correlation_r": None if np.isnan(r) else round(r,3),
            "trend": trend}

def q4_policy(con, state, crop_a, crop_b, n_years=10):
    years = [r[0] for r in con.sql(
        "SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC",[state]).fetchall()][:n_years]
    if not years: return {"error":"No data for region."}
    years_sorted = sorted(years)
    def stats_for(crop):
        df = con.sql(f"""
            SELECT year, SUM(prod_ton) AS prod, SUM(area_ha) AS area
            FROM v_state_crop_year
            WHERE state=? AND crop LIKE ? AND year IN ({",".join(["?"]*len(years_sorted))})
            GROUP BY year ORDER BY year
        """,[state,f"%{crop}%",*years_sorted]).df()
        if df.empty: return None
        df["yield"] = df["prod"]/df["area"].replace(0,np.nan)
        rain = con.sql(f"""
            SELECT year, AVG(annual_mm) AS rain
            FROM fact_rain
            WHERE state=? AND year IN ({",".join(["?"]*len(years_sorted))})
            GROUP BY year ORDER BY year
        """,[state,*years_sorted]).df()
        m = df.merge(rain,on="year",how="inner")
        cov = float(m["prod"].std()/m["prod"].mean()) if m["prod"].mean() else None
        corr = float(np.corrcoef(m["prod"], m["rain"])[0,1]) if len(m)>=3 else None
        y_med = float(np.nanmedian(m["yield"])) if "yield" in m.columns else None
        return {"cov":cov,"corr":corr,"yield_med":y_med}
    a = stats_for(crop_a); b = stats_for(crop_b)
    bullets=[]
    if a and b:
        if a["cov"] is not None and b["cov"] is not None:
            better = crop_a if a["cov"]<b["cov"] else crop_b
            bullets.append(f"Stability (CoV): {better} is less volatile ({a['cov']:.2f} vs {b['cov']:.2f}).")
        if a["corr"] is not None and b["corr"] is not None:
            better = crop_a if abs(a["corr"])<abs(b["corr"]) else crop_b
            bullets.append(f"Rainfall sensitivity: {better} has lower |r| (|r| {abs(a['corr']):.2f} vs {abs(b['corr']):.2f}).")
        if a["yield_med"] and b["yield_med"]:
            better = crop_a if a["yield_med"]>b["yield_med"] else crop_b
            bullets.append(f"Yield median: {better} is higher over last {len(years_sorted)} yrs.")
    return {"arguments": bullets}

# ---------- SIMPLE ROUTER ----------
def detect_intent(q):
    q = (q or "").upper()
    if any(k in q for k in ["HIGHEST","LOWEST","DISTRICT"]): return "Q2"
    if any(k in q for k in ["TREND","CORRELAT","IMPACT"]): return "Q3"
    if any(k in q for k in ["POLICY","PROMOTE","SCHEME"]): return "Q4"
    if any(k in q for k in ["RAINFALL","TOP","COMPARE"]): return "Q1"
    return "Q1"

def parse_two_states(q, sx_default, sy_default):
    qU = q.upper()
    m = re.search(r"IN\s+([A-Z\s&]+?)\s+(?:AND|VS)\s+([A-Z\s&]+)\b", qU)
    if m: return m.group(1).strip(), m.group(2).strip()
    return sx_default, sy_default

def parse_ints(q, default_n=5, default_m=3):
    nums = [int(x) for x in re.findall(r"\b\d+\b", q)]
    N = nums[0] if nums else default_n
    M = nums[1] if len(nums)>1 else default_m
    return max(1,min(N,20)), max(1,min(M,10))

def parse_crop(q, fallback="RICE"):
    m = re.search(r"\b(RICE|WHEAT|SUGARCANE|MAIZE|BAJRA|COTTON|PULSES|MILLET[S]?)\b", q.upper())
    return m.group(1) if m else fallback

# ---------- UI ----------
st.title("Project Samarth — Agri × Climate Q&A (Fast MVP)")
st.caption("Answers 4 required tasks using APY (production) + IMD (rainfall) for 1997–2017. No external services.")

with st.sidebar:
    st.header("Data Inputs")
    use_local = st.toggle("Use local CSVs in repo", value=True)
    apy_file = os.path.join(DATA_DIR, APY_FILE) if use_local else st.file_uploader("Upload APY.csv", type=["csv"])
    imd_file = os.path.join(DATA_DIR, IMD_FILE) if use_local else st.file_uploader("Upload Sub_Division_IMD_2017.csv", type=["csv"])

    st.header("Defaults")
    state_x = st.text_input("State X", "KARNATAKA").upper().strip()
    state_y = st.text_input("State Y", "MAHARASHTRA").upper().strip()
    crop_typ = st.text_input("Crop filter (contains)", "RICE").upper().strip()
    years_n = st.number_input("N years (Q1/Q4)", 3, 20, 5, 1)
    top_m = st.number_input("Top M crops (Q1)", 1, 10, 3, 1)
    region = st.text_input("Region/State (Q3/Q4)", value=state_x).upper().strip()
    crop_a = st.text_input("Policy: Crop A (e.g., BAJRA)", "BAJRA").upper().strip()
    crop_b = st.text_input("Policy: Crop B (e.g., SUGARCANE)", "SUGARCANE").upper().strip()

# load data
if (use_local and (not os.path.exists(apy_file) or not os.path.exists(imd_file))) or ((not use_local) and (apy_file is None or imd_file is None)):
    st.warning("Provide both APY.csv and Sub_Division_IMD_2017.csv to proceed.")
    st.stop()

with st.status("Loading & building in-memory database…", expanded=False):
    apy_df = load_apy(apy_file if use_local else apy_file)
    imd_df = load_imd(imd_file if use_local else imd_file)
    con = build_db(apy_df, imd_df)

st.success("Data ready (1997–2017).")

# Preset buttons for the 4 tasks
st.subheader("Quick Tasks (one-click)")
c1,c2,c3,c4 = st.columns(4)
if c1.button("Q1: Rainfall + Top Crops", key="btn_q1"):
    res = q1_rainfall_and_top_crops(con, state_x, state_y, crop_typ, int(years_n), int(top_m))
    if "error" in res:
        st.warning(res["error"])
    else:
        st.write(f"**Years considered:** {res['years']}")
        st.write(f"**Average rainfall (mm):** {res['avg_rainfall_mm']}")
        st.write(f"**Top {top_m} crops containing '{crop_typ}':**")
        st.write({state_x: res["top_crops"].get(state_x, []), state_y: res["top_crops"].get(state_y, [])})
        st.caption("Sources: APY (production); IMD (state-year rainfall).")

if c2.button("Q2: Highest vs Lowest District", key="btn_q2"):
    res = q2_highest_lowest_district(con, crop_typ, state_x, state_y)
    if "error" in res:
        st.warning(res["error"])
    else:
        st.write(f"**Most recent common year:** {res['year']}")
        st.write(f"**Highest in {state_x}:** {res['highest_in_state_x']}")
        st.write(f"**Lowest in {state_y}:** {res['lowest_in_state_y']}")
        st.caption("Source: APY district records.")

if c3.button("Q3: Trend + Correlation", key="btn_q3"):
    res = q3_trend_and_correlation(con, region, crop_typ, horizon=10)
    if "error" in res:
        st.warning(res["error"])
    else:
        chart_df = pd.DataFrame({"year":res["years"], "production_ton":res["production"], "rainfall_mm":res["rainfall_mm"]})
        st.write(f"Trend: **{res['trend']}**, Correlation r (prod vs rainfall): **{res['correlation_r']}**")
        st.line_chart(chart_df.set_index("year"))
        st.caption("Sources: APY (prod); IMD (rainfall).")

if c4.button("Q4: Policy Argument A vs B", key="btn_q4"):
    res = q4_policy(con, region, crop_a, crop_b, n_years=int(years_n))
    if "error" in res:
        st.warning(res["error"])
    else:
        bullets = res["arguments"] or ["Not enough data overlap to compare."]
        st.markdown("\n".join([f"- {b}" for b in bullets]))
        st.caption("Sources: APY (yield via prod/area); IMD (rainfall).")

st.markdown("---")

# Minimal "chat-like" input with simple routing
st.subheader("Ask in plain English (lightweight routing)")
u_q = st.text_input("Example: Compare rainfall and top 3 rice crops for Karnataka and Maharashtra for last 5 years.")
if u_q:
    intent = detect_intent(u_q)
    N, M = parse_ints(u_q, years_n, top_m)
    sx, sy = parse_two_states(u_q, state_x, state_y)
    crop_q = parse_crop(u_q, crop_typ)

    if intent == "Q1":
        res = q1_rainfall_and_top_crops(con, sx, sy, crop_q, int(N), int(M))
        if "error" in res:
            st.warning(res["error"])
        else:
            st.write(f"**Q1 — Rainfall & Top-{M} crops containing '{crop_q}'**")
            st.write(f"Years: {res['years']}")
            st.write(f"Avg rainfall (mm): {res['avg_rainfall_mm']}")
            st.write({sx: res['top_crops'].get(sx, []), sy: res['top_crops'].get(sy, [])})
            st.caption("Sources: APY; IMD.")

    elif intent == "Q2":
        res = q2_highest_lowest_district(con, crop_q, sx, sy)
        if "error" in res:
            st.warning(res["error"])
        else:
            st.write(f"**Q2 — Highest & Lowest Districts for '{crop_q}' (year {res['year']})**")
            st.write(f"Highest in {sx}: {res['highest_in_state_x']}")
            st.write(f"Lowest in {sy}: {res['lowest_in_state_y']}")
            st.caption("Source: APY.")

    elif intent == "Q3":
        res = q3_trend_and_correlation(con, region, crop_q, horizon=10)
        if "error" in res:
            st.warning(res["error"])
        else:
            st.write(f"**Q3 — Trend & Correlation for '{crop_q}' in {region} (last {len(res['years'])} years)**")
            chart_df = pd.DataFrame({"year":res["years"], "production_ton":res["production"], "rainfall_mm":res["rainfall_mm"]})
            st.line_chart(chart_df.set_index("year"))
            st.write(f"Trend: **{res['trend']}**, r: **{res['correlation_r']}**")
            st.caption("Sources: APY; IMD.")

    else:  # Q4
        res = q4_policy(con, region, crop_q, crop_b, n_years=int(N))
        if "error" in res:
            st.warning(res["error"])
        else:
            bullets = res["arguments"] or ["Not enough data overlap to compare."]
            st.write(f"**Q4 — Policy Argument: Promote {crop_q} vs {crop_b} in {region} (last {N} years)**")
            st.markdown("\n".join([f"- {b}" for b in bullets]))
            st.caption("Sources: APY (yield via prod/area); IMD (rainfall).")

st.markdown("---")
st.subheader("Citations")
st.write("""
- **APY (MoA&FW/DES)**: District-wise, season-wise crop production statistics (1997–2017 slice).
- **IMD (MoES)**: Sub-division rainfall aggregated to state-year annual mm (up to 2017).
- **Method note**: Rainfall per state approximated as mean of constituent IMD sub-divisions in this prototype.
""")
