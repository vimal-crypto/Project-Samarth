import duckdb
import numpy as np
import pandas as pd

def build_db(crop_df: pd.DataFrame, rain_df: pd.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("crop_df", crop_df)
    con.register("rain_df", rain_df)
    con.execute("""CREATE TABLE fact_crop AS
        SELECT UPPER(TRIM(state)) AS state,
               UPPER(TRIM(district)) AS district,
               UPPER(TRIM(crop)) AS crop,
               COALESCE(UPPER(TRIM(season)),'WHOLE YEAR') AS season,
               CAST(year AS INTEGER) AS year,
               CAST(area_ha AS DOUBLE) AS area_ha,
               CAST(production_ton AS DOUBLE) AS production_ton
        FROM crop_df WHERE year IS NOT NULL
    """)
    con.execute("""CREATE TABLE fact_rain AS
        SELECT UPPER(TRIM(state)) AS state,
               CAST(year AS INTEGER) AS year,
               CAST(annual_mm AS DOUBLE) AS annual_mm
        FROM rain_df WHERE year IS NOT NULL
    """)
    con.execute("""CREATE VIEW v_state_crop_year AS
        SELECT state, crop, year,
               SUM(production_ton) AS prod_ton,
               SUM(area_ha) AS area_ha
        FROM fact_crop GROUP BY state, crop, year
    """)
    con.execute("""CREATE VIEW v_state_year AS
        SELECT state, year,
               SUM(production_ton) AS prod_ton,
               SUM(area_ha) AS area_ha
        FROM fact_crop GROUP BY state, year
    """)
    return con

def q1_rainfall_and_top_crops(con, state_x, state_y, crop_like, n_years, top_m):
    years_x = con.execute(
        "SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC", [state_x]
    ).fetchall()
    years_y = con.execute(
        "SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC", [state_y]
    ).fetchall()
    years_x = [y[0] for y in years_x]; years_y = [y[0] for y in years_y]
    years = [y for y in years_x if y in years_y][:n_years]
    if not years: return {"error":"No overlapping years."}

    q = f"SELECT state, year, annual_mm FROM fact_rain WHERE state IN (?,?) AND year IN ({','.join(['?']*len(years))})"
    rain = con.execute(q, [state_x, state_y, *years]).fetchdf()
    avg_rain = rain.groupby("state")["annual_mm"].mean().round(1).to_dict()

    q = f"""SELECT state, crop, SUM(prod_ton) AS prod
            FROM v_state_crop_year
            WHERE state IN (?,?) AND year IN ({','.join(['?']*len(years))})
              AND crop LIKE ?
            GROUP BY state, crop ORDER BY state, prod DESC"""
    crops = con.execute(q, [state_x, state_y, *years, f"%{crop_like}%"]).fetchdf()
    topx = crops[crops["state"]==state_x].nlargest(top_m,"prod")
    topy = crops[crops["state"]==state_y].nlargest(top_m,"prod")
    return {"years": years, "avg_rainfall_mm": avg_rain,
            "top_crops": {state_x: topx[["crop","prod"]].values.tolist(),
                          state_y: topy[["crop","prod"]].values.tolist()}}

def q2_highest_lowest_district(con, crop_like, state_x, state_y):
    yx = con.execute("SELECT MAX(year) FROM fact_crop WHERE state=?", [state_x]).fetchone()[0]
    yy = con.execute("SELECT MAX(year) FROM fact_crop WHERE state=?", [state_y]).fetchone()[0]
    if not yx or not yy: return {"error":"No recent year for one/both states."}
    year = min(yx, yy)
    df = con.execute("""SELECT state, district, SUM(production_ton) AS prod
                        FROM fact_crop WHERE year=? AND crop LIKE ?
                        GROUP BY state, district""", [year, f"%{crop_like}%"]).fetchdf()
    hi_x = df[df["state"]==state_x].nlargest(1,"prod")
    lo_y = df[df["state"]==state_y].nsmallest(1,"prod")
    return {"year":year, "highest_in_state_x":hi_x.values.tolist(), "lowest_in_state_y":lo_y.values.tolist()}

def q3_trend_and_correlation(con, state, crop_like, horizon=10):
    years = con.execute("SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC", [state]).fetchall()
    years = [y[0] for y in years][:horizon]; years_sorted = sorted(years)
    if not years_sorted: return {"error":"No years for region."}
    q = f"""SELECT year, SUM(prod_ton) AS prod FROM v_state_crop_year
            WHERE state=? AND crop LIKE ? AND year IN ({','.join(['?']*len(years_sorted))})
            GROUP BY year ORDER BY year"""
    prod = con.execute(q, [state, f"%{crop_like}%", *years_sorted]).fetchdf()
    q = f"""SELECT year, AVG(annual_mm) AS rain FROM fact_rain
            WHERE state=? AND year IN ({','.join(['?']*len(years_sorted))})
            GROUP BY year ORDER BY year"""
    rain = con.execute(q, [state, *years_sorted]).fetchdf()
    m = prod.merge(rain, on="year", how="inner")
    if len(m)>=3:
        r = float(np.corrcoef(m["prod"], m["rain"])[0,1])
    else:
        r = float("nan")
    trend = "increasing" if m["prod"].diff().mean()>0 else "decreasing"
    return {"years":m["year"].tolist(), "production":m["prod"].round(2).tolist(),
            "rainfall_mm":m["rain"].round(1).tolist(), "correlation_r":None if np.isnan(r) else round(r,3),
            "trend":trend}

def q4_policy(con, state, crop_a, crop_b, n_years=7):
    years = con.execute("SELECT DISTINCT year FROM v_state_crop_year WHERE state=? ORDER BY year DESC", [state]).fetchall()
    years = [y[0] for y in years][:n_years]; years_sorted = sorted(years)
    if not years_sorted: return {"error":"No data for region."}

    def stats_for(c):
        q = f"""SELECT year, SUM(prod_ton) AS prod, SUM(area_ha) AS area
                FROM v_state_crop_year WHERE state=? AND crop LIKE ? AND year IN ({','.join(['?']*len(years_sorted))})
                GROUP BY year ORDER BY year"""
        df = con.execute(q, [state, f"%{c}%", *years_sorted]).fetchdf()
        if df.empty: return None
        df["yield"] = df["prod"]/df["area"].replace(0,np.nan)
        q = f"""SELECT year, AVG(annual_mm) AS rain FROM fact_rain
                WHERE state=? AND year IN ({','.join(['?']*len(years_sorted))})
                GROUP BY year ORDER BY year"""
        rain = con.execute(q, [state, *years_sorted]).fetchdf()
        m = df.merge(rain, on="year", how="inner")
        cov = float(m["prod"].std()/m["prod"].mean()) if m["prod"].mean() else None
        corr = float(np.corrcoef(m["prod"], m["rain"])[0,1]) if len(m)>=3 else None
        y_med = float(np.nanmedian(m["yield"])) if "yield" in m.columns else None
        return {"cov":cov,"corr":corr,"yield_med":y_med}

    A = stats_for(crop_a); B = stats_for(crop_b); bullets=[]
    if A and B:
        if A["cov"] is not None and B["cov"] is not None:
            better = crop_a if A["cov"]<B["cov"] else crop_b
            bullets.append(f"Stability (CoV): {better} is less volatile.")
        if A["corr"] is not None and B["corr"] is not None:
            better = crop_a if abs(A["corr"])<abs(B["corr"]) else crop_b
            bullets.append(f"Lower rainfall sensitivity: {better} has lower |r|.")
        if A["yield_med"] and B["yield_med"]:
            better = crop_a if A["yield_med"]>B["yield_med"] else crop_b
            bullets.append(f"Higher median yield: {better} over last {len(years_sorted)} yrs.")
    return {"arguments": bullets}
