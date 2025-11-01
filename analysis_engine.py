# analysis_engine.py (Python 3.9 safe)
import numpy as np
import duckdb
import pandas as pd

def build_db(crop_df: pd.DataFrame, rain_df: pd.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("crop_df", crop_df)
    con.register("rain_df", rain_df)

    con.execute("DROP TABLE IF EXISTS fact_crop")
    con.execute("""
        CREATE TABLE fact_crop AS
        SELECT
          UPPER(TRIM(state))    AS state,
          UPPER(TRIM(district)) AS district,
          UPPER(TRIM(crop))     AS crop,
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

    con.execute("""
        CREATE OR REPLACE VIEW v_state_crop_year AS
        SELECT state, crop, year,
               SUM(production_ton) AS prod_ton,
               SUM(area_ha) AS area_ha
        FROM fact_crop
        GROUP BY state, crop, year
    """)
    con.execute("""
        CREATE OR REPLACE VIEW v_state_year AS
        SELECT state, year,
               SUM(production_ton) AS prod_ton,
               SUM(area_ha) AS area_ha
        FROM fact_crop
        GROUP BY state, year
    """)
    return con

def q1_rainfall_and_top_crops(con, state_x, state_y, crop_filter, n_years, top_m):
    years_x = [r[0] for r in con.sql(
        "SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC",
        [state_x]).fetchall()]
    years_y = [r[0] for r in con.sql(
        "SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC",
        [state_y]).fetchall()]
    years = [y for y in years_x if y in years_y][:n_years]
    if not years:
        return {"error": "No overlapping years for those states."}

    rain = con.sql(f"""
        SELECT state, year, annual_mm
        FROM fact_rain
        WHERE state IN (?,?) AND year IN ({",".join(["?"]*len(years))})
    """, [state_x, state_y, *years]).df()
    avg_rain = rain.groupby("state")["annual_mm"].mean().round(1).to_dict()

    crops = con.sql(f"""
        SELECT state, crop, SUM(prod_ton) AS prod
        FROM v_state_crop_year
        WHERE state IN (?,?) AND year IN ({",".join(["?"]*len(years))})
          AND crop LIKE ?
        GROUP BY state, crop
        ORDER BY state, prod DESC
    """, [state_x, state_y, *years, f"%{crop_filter}%"]).df()

    topx = crops[crops["state"]==state_x].nlargest(top_m, "prod")
    topy = crops[crops["state"]==state_y].nlargest(top_m, "prod")
    return {
        "years": years,
        "avg_rainfall_mm": avg_rain,
        "top_crops": {
            state_x: topx[["crop","prod"]].values.tolist(),
            state_y: topy[["crop","prod"]].values.tolist()
        }
    }

def q2_highest_lowest_district(con, crop_z, state_x, state_y):
    yx = con.sql("SELECT MAX(year) FROM fact_crop WHERE state=?", [state_x]).fetchone()[0]
    yy = con.sql("SELECT MAX(year) FROM fact_crop WHERE state=?", [state_y]).fetchone()[0]
    if not yx or not yy:
        return {"error": "No recent year found for one or both states."}
    year = min(yx, yy)
    df = con.sql("""
        SELECT state, district, SUM(production_ton) AS prod
        FROM fact_crop
        WHERE year=? AND crop LIKE ?
        GROUP BY state, district
    """, [year, f"%{crop_z}%"]).df()
    hi_x = df[df["state"]==state_x].nlargest(1, "prod")
    lo_y = df[df["state"]==state_y].nsmallest(1, "prod")
    return {
        "year": year,
        "highest_in_state_x": hi_x.values.tolist(),
        "lowest_in_state_y": lo_y.values.tolist()
    }

def q3_trend_and_correlation(con, state, crop_c, horizon=10):
    years = [r[0] for r in con.sql(
        "SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC", [state]
    ).fetchall()][:horizon]
    years_sorted = sorted(years)
    if not years_sorted:
        return {"error": "No years for region."}

    prod = con.sql(f"""
        SELECT year, SUM(prod_ton) AS prod
        FROM v_state_crop_year
        WHERE state=? AND crop LIKE ? AND year IN ({",".join(["?"]*len(years_sorted))})
        GROUP BY year ORDER BY year
    """, [state, f"%{crop_c}%", *years_sorted]).df()

    rain = con.sql(f"""
        SELECT year, AVG(annual_mm) AS rain
        FROM fact_rain
        WHERE state=? AND year IN ({",".join(["?"]*len(years_sorted))})
        GROUP BY year ORDER BY year
    """, [state, *years_sorted]).df()

    m = prod.merge(rain, on="year", how="inner")
    if len(m) >= 3:
        r = float(np.corrcoef(m["prod"], m["rain"])[0,1])
    else:
        r = float("nan")
    trend = "increasing" if m["prod"].diff().mean() > 0 else "decreasing"

    return {
        "years": m["year"].tolist(),
        "production": m["prod"].round(2).tolist(),
        "rainfall_mm": m["rain"].round(1).tolist(),
        "correlation_r": None if np.isnan(r) else round(r, 3),
        "trend": trend
    }

def q4_policy(con, state, crop_a, crop_b, n_years=10):
    years = [r[0] for r in con.sql(
        "SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC", [state]
    ).fetchall()][:n_years]
    if not years:
        return {"error": "No data for region."}
    years_sorted = sorted(years)

    def stats_for(crop):
        df = con.sql(f"""
            SELECT year, SUM(prod_ton) AS prod, SUM(area_ha) AS area
            FROM v_state_crop_year
            WHERE state=? AND crop LIKE ? AND year IN ({",".join(["?"]*len(years_sorted))})
            GROUP BY year ORDER BY year
        """, [state, f"%{crop}%", *years_sorted]).df()
        if df.empty:
            return None
        df["yield"] = df["prod"]/df["area"].replace(0, np.nan)
        rain = con.sql(f"""
            SELECT year, AVG(annual_mm) AS rain
            FROM fact_rain
            WHERE state=? AND year IN ({",".join(["?"]*len(years_sorted))})
            GROUP BY year ORDER BY year
        """, [state, *years_sorted]).df()
        m = df.merge(rain, on="year", how="inner")
        cov = float(m["prod"].std()/m["prod"].mean()) if m["prod"].mean() else None
        corr = float(np.corrcoef(m["prod"], m["rain"])[0,1]) if len(m)>=3 else None
        y_med = float(np.nanmedian(m["yield"])) if "yield" in m.columns else None
        return {"cov":cov, "corr":corr, "yield_med":y_med}

    a = stats_for(crop_a)
    b = stats_for(crop_b)

    bullets = []
    if a and b:
        if a["cov"] is not None and b["cov"] is not None:
            better = crop_a if a["cov"] < b["cov"] else crop_b
            bullets.append(f"Stability (CoV): {better} is less volatile ({a['cov']:.2f} vs {b['cov']:.2f}).")
        if a["corr"] is not None and b["corr"] is not None:
            better = crop_a if abs(a["corr"]) < abs(b["corr"]) else crop_b
            bullets.append(f"Rainfall sensitivity: {better} has lower |r| (|r| {abs(a['corr']):.2f} vs {abs(b['corr']):.2f}).")
        if a["yield_med"] and b["yield_med"]:
            better = crop_a if a["yield_med"] > b["yield_med"] else crop_b
            bullets.append(f"Median yield: {better} is higher over last {len(years_sorted)} yrs.")
    return {"arguments": bullets}
