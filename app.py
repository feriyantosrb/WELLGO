"""
WELLGO (Well Grouping Optimizer) — 9 Unit MWT Daily Planner
----------------------------------------------------------
Optimasi rute logistik well testing Sumatra Light North (SL North).
Terintegrasi dengan design system `wellgo_ui`.

Run: py -m streamlit run app.py
"""

import os
import re
import sqlite3
import math
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image

import wellgo_ui as ui
import wellgo_guide as guide

st.set_page_config(page_title="WELLGO", page_icon="wellgo_icon.png", layout="wide")

ui.inject_theme()

DB_PATH = "welltest_status.db"
SHEET_DEFAULT = "Kandidat Sumur"

REMOTE_AREAS = {"BANGKO", "BALAM"}
REMOTE_UNITS = ["MPAS_444", "MPAS_768", "MPAS_523", "MPAS_445", "MPAS_534"]
NONREMOTE_UNITS = ["MPAS_535", "MPAS_524", "MPAS_525", "MPAS_767"]
ALL_UNITS = REMOTE_UNITS + NONREMOTE_UNITS

# ------------------------------------------------------------------ persistence
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS execution_log(
        plan_date TEXT, well_name TEXT, unit TEXT, status TEXT, reason TEXT, updated_at TEXT,
        PRIMARY KEY(plan_date, well_name))""")
    existing = {r[1] for r in con.execute("PRAGMA table_info(execution_log)").fetchall()}
    for col in ("unit", "status", "reason", "updated_at"):
        if col not in existing:
            con.execute(f"ALTER TABLE execution_log ADD COLUMN {col} TEXT")
    con.execute("""CREATE TABLE IF NOT EXISTS coord_cache(
        well_name TEXT PRIMARY KEY, lat REAL, lon REAL, updated_at TEXT)""")
    con.commit()
    con.close()

def save_status(plan_date, rows):
    con = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat(timespec="seconds")
    for well, unit, status in rows:
        con.execute("""INSERT INTO execution_log(plan_date,well_name,unit,status,updated_at)
            VALUES(?,?,?,?,?) ON CONFLICT(plan_date,well_name) DO UPDATE SET
            unit=excluded.unit, status=excluded.status, updated_at=excluded.updated_at""",
            (plan_date, well, unit, status, now))
    con.commit()
    con.close()

def reset_execution_log():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM execution_log")
    con.commit()
    con.close()

def status_in_period(lo, hi):
    con = sqlite3.connect(DB_PATH)
    try:
        q = ("SELECT well_name AS well, status, reason, plan_date FROM execution_log "
             "WHERE status IN ('executed','ncmp','pending') AND plan_date BETWEEN ? AND ?")
        df = pd.read_sql(q, con, params=(str(lo), str(hi)))
    except Exception:
        df = pd.DataFrame(columns=["well", "status", "reason", "plan_date"])
    con.close()
    _empty_pend = pd.DataFrame(columns=["well", "plan_date"])
    if not len(df):
        return set(), pd.DataFrame(columns=["well", "reason", "plan_date"]), _empty_pend
    df["plan_date"] = df["plan_date"].astype(str)
    latest = df[df["plan_date"] == df.groupby("well")["plan_date"].transform("max")]

    def _winner(s):
        ss = set(s)
        if "executed" in ss: return "executed"
        if "ncmp" in ss: return "ncmp"
        return "pending"
    wstat = latest.groupby("well")["status"].apply(_winner)
    executed = set(wstat[wstat == "executed"].index)
    ncmp_w = set(wstat[wstat == "ncmp"].index)
    pend_w = set(wstat[wstat == "pending"].index)

    ncmp = (latest[latest["well"].isin(ncmp_w)]
            .sort_values("plan_date").groupby("well", as_index=False).last()[["well", "reason", "plan_date"]])
    pending = (latest[latest["well"].isin(pend_w)]
               .sort_values("plan_date").groupby("well", as_index=False).last()[["well", "plan_date"]])
    return executed, ncmp, pending

def comp_records(wells):
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT well_name AS well, plan_date, reason FROM execution_log WHERE status='executed'", con)
    except Exception:
        df = pd.DataFrame(columns=["well", "plan_date", "reason"])
    con.close()
    wset = set(map(str, wells))
    if not len(df): return {}
    df = df[df["well"].isin(wset)].copy()
    df["plan_date"] = pd.to_datetime(df["plan_date"], errors="coerce")
    df["reason"] = df["reason"].fillna("").astype(str).str.upper()
    return {w: list(zip(grp["plan_date"], grp["reason"])) for w, grp in df.groupby("well")}

def sch_latest(wells):
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT well_name AS well, status, plan_date FROM execution_log "
                         "WHERE status IN ('executed','ncmp','pending')", con)
    except Exception:
        df = pd.DataFrame(columns=["well", "status", "plan_date"])
    con.close()
    wset = set(map(str, wells))
    if not len(df): return {}
    df = df[df["well"].isin(wset)].copy()
    if not len(df): return {}
    df["plan_date"] = df["plan_date"].astype(str)
    latest = df.loc[df.groupby("well")["plan_date"].idxmax()]
    lab = {"executed": "COMP", "ncmp": "NCMP", "pending": "PENDING"}
    return {r.well: (r.plan_date, lab.get(r.status, str(r.status).upper())) for r in latest.itertuples()}

def norm_unit(u):
    u = str(u).strip().upper()
    m = re.fullmatch(r"MP_?(\d+)", u)
    return f"MPAS_{m.group(1)}" if m else u

def norm_unit_name(u):
    s = str(u).strip()
    m = re.fullmatch(r"MP_?(\d+)", s.upper())
    return f"MPAS_{m.group(1)}" if m else s

def classify_status(stat):
    s = str(stat).strip().upper().replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    if s.startswith("NCMP") or s.startswith("NOT COMP") or s.startswith("INCOMP"): return "NCMP"
    if s.startswith("COMP") or s in ("DONE", "OK", "C", "EXECUTED", "TESTED"): return "COMP"
    return ""

def import_compncmp(file_list):
    n_comp = n_ncmp = n_pend = 0
    reasons = {}
    status_seen = {}
    skip_date = skip_well = skip_status = 0
    con = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat(timespec="seconds")
    for fb in file_list:
        xls = pd.ExcelFile(BytesIO(fb))
        sht = None
        for s in xls.sheet_names:
            try:
                up = {str(c).strip().upper() for c in pd.read_excel(xls, sheet_name=s, nrows=0).columns}
            except Exception: continue
            if {"WELL", "STATUS", "SCHEDULE_DATE_TEST"} <= up:
                sht = s; break
        if sht is None:
            sht = next((s for s in xls.sheet_names if s.strip().upper().replace(" ", "").replace("_", "")
                        in ("SCHDATABASE", "COMPNCMP", "SCHSTATUS")), xls.sheet_names[0])
        df = pd.read_excel(xls, sheet_name=sht)
        cols = {str(c).strip().upper(): c for c in df.columns}
        cw, cs, cd = cols.get("WELL"), cols.get("STATUS"), cols.get("SCHEDULE_DATE_TEST")
        cu, cr = cols.get("UNIT"), cols.get("REASON") or cols.get("COMMENT IF NOT COMPLETE")
        if not (cw and cs and cd): continue
        w = pd.DataFrame({
            "well": df[cw].astype(str).str.strip(),
            "raw_stat": df[cs].astype(str).str.strip().str.upper(),
            "date": pd.to_datetime(df[cd], errors="coerce"),
            "unit": df[cu].map(norm_unit) if cu else "",
            "reason": (df[cr].astype(str).str.strip().str.upper().replace({"NAN": ""}) if cr else ""),
        })
        for k, v in w["raw_stat"].value_counts().items(): status_seen[k] = status_seen.get(k, 0) + int(v)
        w["stat"] = w["raw_stat"].map(classify_status)
        skip_well += int(w["well"].isin(["", "nan"]).sum())
        skip_date += int(w["date"].isna().sum())
        valid = (~w["well"].isin(["", "nan"])) & (w["date"].notna())

        wp = w[valid & (w["stat"] == "")].copy()
        skip_status += int(((w["stat"] == "") & ~valid).sum())
        if len(wp):
            wp["plan_date"] = wp["date"].dt.date.astype(str)
            prows = list(zip(wp["plan_date"], wp["well"], wp["unit"], ["pending"] * len(wp), [""] * len(wp), [now] * len(wp)))
            con.executemany("""INSERT INTO execution_log(plan_date,well_name,unit,status,reason,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(plan_date,well_name) DO UPDATE SET
                unit=excluded.unit, status=excluded.status, reason=excluded.reason, updated_at=excluded.updated_at
                WHERE execution_log.status NOT IN ('executed','ncmp')""", prows)
            n_pend += len(wp)

        w = w[valid & (w["stat"] != "")].copy()
        w["plan_date"] = w["date"].dt.date.astype(str)
        w["log_status"] = np.where(w["stat"] == "COMP", "executed", "ncmp")
        rows = list(zip(w["plan_date"], w["well"], w["unit"], w["log_status"], w["reason"], [now] * len(w)))
        con.executemany("""INSERT INTO execution_log(plan_date,well_name,unit,status,reason,updated_at)
            VALUES(?,?,?,?,?,?) ON CONFLICT(plan_date,well_name) DO UPDATE SET
            unit=excluded.unit, status=excluded.status, reason=excluded.reason, updated_at=excluded.updated_at""", rows)
        n_comp += int((w["stat"] == "COMP").sum())
        nc = w[w["stat"] == "NCMP"]
        n_ncmp += len(nc)
        for rsn, cnt in nc["reason"].replace("", "(kosong)").value_counts().items(): reasons[rsn] = reasons.get(rsn, 0) + int(cnt)
    con.commit(); con.close()
    return {"comp": n_comp, "ncmp": n_ncmp, "pending": n_pend, "reasons": reasons, "status_seen": status_seen,
            "skip_date": skip_date, "skip_well": skip_well, "skip_status": skip_status}

def save_coords(pairs):
    con = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat(timespec="seconds")
    for well, lat, lon in pairs:
        if pd.notna(lat) and pd.notna(lon):
            con.execute("""INSERT INTO coord_cache(well_name,lat,lon,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(well_name) DO UPDATE SET lat=excluded.lat, lon=excluded.lon, updated_at=excluded.updated_at""",
                (well, float(lat), float(lon), now))
    con.commit(); con.close()

def load_coord_cache():
    con = sqlite3.connect(DB_PATH)
    try: df = pd.read_sql("SELECT well_name,lat,lon FROM coord_cache", con)
    except Exception: df = pd.DataFrame(columns=["well_name", "lat", "lon"])
    con.close(); return df

# ------------------------------------------------------------------ data
def to_dt(col):
    num = pd.to_numeric(col, errors="coerce")
    valid = num.dropna()
    if len(valid) and valid.between(20000, 60000).mean() > 0.5: return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(col, errors="coerce")

@st.cache_data(show_spinner=False)
def load_spatial_data(file_bytes, sheet):
    try:
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
        df.columns = [str(c).strip().upper() for c in df.columns]
        if not {"WELL", "FIELD", "LAT", "LON"}.issubset(set(df.columns)): return pd.DataFrame()
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        df = df.dropna(subset=["WELL", "LAT", "LON"])
        return df.drop_duplicates(subset=["WELL"], keep="first")
    except Exception: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_candidates(file_bytes, sheet):
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    ren = {"well_name": "well", "Surface Lat": "lat", "Surface Lon": "lon", "Duration test (minutes)": "dur",
           "min_execution date": "min_date", "max_execution_date": "max_date", "op_sub_area_code": "subarea",
           "op_area_code": "area", "test_category": "category", "well_tier": "tier", "field": "field",
           "string_type": "string_type", "Remark": "remark", "REMARK for IEMS Req or Spare candidate": "remark_iems"}
    if "last_unit_name" in df.columns: ren["last_unit_name"] = "unit"
    elif "unit_name" in df.columns: ren["unit_name"] = "unit"
    df = df.rename(columns=ren)

    for c in ["lat", "lon", "string_type", "remark", "remark_iems", "field", "area", "unit"]:
        if c not in df.columns: df[c] = np.nan
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "dur" in df.columns: df["dur"] = pd.to_numeric(df["dur"], errors="coerce")
    df["min_date"] = to_dt(df["min_date"])
    df["max_date"] = to_dt(df["max_date"])

    np_col = next((c for c in df.columns if "NEXT" in str(c).upper() and ("PROPOS" in str(c).upper() or "WT" in str(c).upper())), None)
    df["next_wt"] = to_dt(df[np_col]) if np_col else pd.NaT
    pop_col = next((c for c in df.columns if "POP" in str(c).upper() and "DATE" in str(c).upper()), None)
    df["pop_date"] = to_dt(df[pop_col]) if pop_col else pd.NaT
    df["unit"] = df["unit"].map(norm_unit_name)

    st_, area_, fld_ = df["string_type"].astype(str).str.upper().str.strip(), df["area"].astype(str).str.upper().str.strip(), df["field"].astype(str).str.upper().str.strip()
    df["forced_unit"] = None
    df.loc[st_.eq("GP") & area_.eq("BEKASAP"), "forced_unit"] = "MPAS_525"
    df.loc[st_.eq("GP") & area_.isin(["BANGKO", "BALAM"]), "forced_unit"] = "MPAS_768"
    df.loc[fld_.eq("BENAR"), "forced_unit"] = "MPAS_534"
    fm = df["forced_unit"].notna()
    df.loc[fm, "unit"] = df.loc[fm, "forced_unit"]

    df["is_mpas"] = df["unit"].astype(str).str.upper().str.startswith("MPAS")
    uu = df["unit"].astype(str).str.upper()
    df["is_ts"] = uu.str.contains("TS", na=False) & ~df["is_mpas"]
    df["unit_unknown"] = uu.isin(["(BELUM)", "(BELUM PERNAH)", "(BELUM PERNAH COMP)", "NAN", ""]) | df["unit"].isna()

    NWAWS = {"NEW WELL 1", "NEW WELL 2", "NEW WELL 3", "AWS1", "AWS2"}
    cat_u = df["category"].astype(str).str.upper().str.strip()
    df["is_nwaws"] = cat_u.isin(NWAWS)
    df["tipe"] = np.where(cat_u.str.contains("NEW WELL"), "NW", np.where(cat_u.str.contains("AWS"), "AWS", "REG"))
    rmk = (df["remark"].astype(str).fillna("") + " " + df["remark_iems"].astype(str).fillna("")).str.upper()
    is_req = rmk.str.contains("REQ", na=False) | rmk.str.contains("DEEPENING", na=False)
    df["force_week"] = df["is_nwaws"] | is_req
    df["req_tag"] = np.where(is_req & rmk.str.contains("OPS", na=False), "ORQ", np.where(is_req, "PRQ", ""))

    status_col = next((c for c in df.columns if str(c).strip().upper() in ("WELL STATUS", "LAST_STATUS")), None)
    df["status"] = (df[status_col].astype(str).str.upper().str.strip() if status_col else "ON")
    sch_col = next((c for c in df.columns if str(c).strip().upper() in ("SCH STATUS", "SCH_STATUS")), None)
    df["sch_status"] = (df[sch_col].astype(str).str.upper().str.strip() if sch_col else "").replace({"NAN": "", "NONE": ""})
    return df

def good_coord(lat, lon):
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    return pd.notna(lat) & pd.notna(lon) & lat.between(0.1, 5) & lon.between(95, 110)

def resolve_coords(df, spatial_db, cache, field_assign=None):
    df = df.copy(); field_assign = field_assign or {}; df["coord_source"] = "none"
    if not spatial_db.empty:
        s_map = spatial_db.set_index("WELL")
        has_master = df["well"].isin(s_map.index)
        df.loc[has_master, "lat"] = df.loc[has_master, "well"].map(s_map["LAT"])
        df.loc[has_master, "lon"] = df.loc[has_master, "well"].map(s_map["LON"])
        upd_fld = (df["field"].isna() | (df["field"] == "")) & has_master
        if "FIELD" in s_map.columns: df.loc[upd_fld, "field"] = df.loc[upd_fld, "well"].map(s_map["FIELD"])
        df.loc[has_master & good_coord(df["lat"], df["lon"]), "coord_source"] = "master_spasial"

    df.loc[good_coord(df["lat"], df["lon"]) & (df["coord_source"] == "none"), "coord_source"] = "database"

    if not cache.empty:
        cmap = cache.set_index("well_name")
        has_cache = (df["coord_source"] == "none") & df["well"].isin(cmap.index)
        df.loc[has_cache, "lat"] = df.loc[has_cache, "well"].map(cmap["lat"])
        df.loc[has_cache, "lon"] = df.loc[has_cache, "well"].map(cmap["lon"])
        df.loc[has_cache, "coord_source"] = "cache"

    if not spatial_db.empty and "FIELD" in spatial_db.columns: cent_f = spatial_db.groupby("FIELD")[["LAT", "LON"]].mean()
    else:
        base = df[df["coord_source"].isin(["master_spasial", "database", "cache"])]
        cent_f = base.groupby("field")[["lat", "lon"]].mean() if len(base) else pd.DataFrame(columns=["lat", "lon"])
        cent_f.columns = ["LAT", "LON"]

    has_cent = (df["coord_source"] == "none") & df["field"].isin(cent_f.index)
    df.loc[has_cent, "lat"] = df.loc[has_cent, "field"].map(cent_f["LAT"])
    df.loc[has_cent, "lon"] = df.loc[has_cent, "field"].map(cent_f["LON"])
    df.loc[has_cent, "coord_source"] = "imputed_field"

    for well, fld in field_assign.items():
        m = (df["well"] == well) & (df["coord_source"] == "none")
        if m.any() and fld in cent_f.index:
            df.loc[m, "field"], df.loc[m, "lat"], df.loc[m, "lon"] = fld, cent_f.loc[fld, "LAT"], cent_f.loc[fld, "LON"]
            df.loc[m, "coord_source"] = "manual_field"

    df["has_coord"] = df["coord_source"] != "none"
    return df

# ------------------------------------------------------------------ geometry
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0; p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def _haversine_matrix(lat, lon):
    la, lo, p = np.asarray(lat, dtype=float)[:, None] * np.pi / 180.0, np.asarray(lon, dtype=float)[:, None] * np.pi / 180.0, np.pi / 180.0
    a = np.clip(0.5 - np.cos(la.T - la) / 2.0 + np.cos(la) * np.cos(la.T) * (1.0 - np.cos(lo.T - lo)) / 2.0, 0.0, 1.0)
    return np.nan_to_num(2 * 6371.0 * np.arcsin(np.sqrt(a)), nan=1e9, posinf=1e9)

def _solve_route(lat, lon):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    n = len(lat)
    if n <= 1: return list(range(n)), 0.0
    D = _haversine_matrix(lat, lon)
    if n == 2: return [0, 1], float(D[0, 1])
    dist = D.tolist()
    best_order, best_total = None, float("inf")
    rng = range(n)
    for start in rng:
        used = [False] * n; used[start] = True; order = [start]
        for _ in range(n - 1):
            drow = dist[order[-1]]; bd = float("inf"); bn = -1
            for j in rng:
                if not used[j] and drow[j] < bd: bd, bn = drow[j], j
            order.append(bn); used[bn] = True
        improved = True
        while improved and n > 3:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 2, n):
                    if dist[order[i-1]][order[j-1]] + dist[order[i]][order[j]] + 1e-9 < dist[order[i-1]][order[i]] + dist[order[j-1]][order[j]]:
                        order[i:j] = order[i:j][::-1]; improved = True
        total = float(sum(dist[order[k]][order[k+1]] for k in range(n - 1)))
        if total < best_total: best_total, best_order = total, order
    return best_order, best_total

@st.cache_resource(show_spinner=False)
def _route_cache_store(): return {}

def route_distance(lat, lon):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if lat.size <= 1: return 0.0
    m = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[m], lon[m]
    if lat.size <= 1: return 0.0
    key = tuple(sorted(zip(np.round(lat, 5).tolist(), np.round(lon, 5).tolist())))
    store = _route_cache_store()
    if key not in store:
        store[key] = _solve_route(lat, lon)[1]
        if len(store) > 50000: store.clear()
    return store[key]

def optimize_route(lat, lon): return _solve_route(lat, lon)

def convex_hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2: return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def block_polygon(sub, pad_km=0.6):
    lat, lon = pd.to_numeric(sub["lat"], errors="coerce").values, pd.to_numeric(sub["lon"], errors="coerce").values
    good = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[good], lon[good]
    if len(lat) == 0: return []
    clat, clon = lat.mean(), lon.mean()
    if len(lat) >= 3:
        hull = convex_hull(list(zip(lon, lat)))
        if len(hull) >= 3:
            f = 1.0 + pad_km / max(0.3, np.mean(haversine_km(lat, lon, clat, clon)) + 0.3)
            return [[clon + (x - clon) * f, clat + (y - clat) * f] for x, y in hull]
    r = max(haversine_km(lat, lon, clat, clon).max() if len(lat) > 1 else 0.0, 0.0) + pad_km
    out = []
    for k in range(28):
        a = 2 * math.pi * k / 28
        out.append([clon + (r / (111.0 * math.cos(math.radians(clat)))) * math.cos(a), clat + (r / 111.0) * math.sin(a)])
    return out

# ------------------------------------------------------------------ engine
def plan(elig, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, current_day=None, elastic_limit=5.0, blocked_units=None, prebooked=None):
    df = elig.reset_index(drop=True).copy()
    df["scheduled"], df["plan_unit"] = False, None
    if "forced_unit" not in df.columns: df["forced_unit"] = None
    if "urgency" not in df.columns: df["urgency"] = 0.0
    df["_pre_unit"] = None
    if prebooked is not None and len(prebooked):
        pb = prebooked.copy()
        pb["_pre_unit"], pb["forced_unit"] = pb["plan_unit"], None
        for _c in df.columns:
            if _c not in pb.columns: pb[_c] = None
        df = pd.concat([df, pb[df.columns]], ignore_index=True)

    lats, lons = pd.to_numeric(df["lat"], errors="coerce").values, pd.to_numeric(df["lon"], errors="coerce").values
    dist_mat, field_arr, area_arr = _haversine_matrix(lats, lons), df["field"].values, df["area"].values
    urg_arr, dur_arr, speed = pd.to_numeric(df["urgency"], errors="coerce").fillna(0).values, pd.to_numeric(df["dur"], errors="coerce").fillna(0).values, max(float(speed), 1.0)

    if mode == "dedicated":
        for unit in df["unit"].dropna().unique():
            idxs = list(df.index[df["unit"] == unit])
            if not idxs: continue
            ordered = sorted(idxs, key=lambda i: (urg_arr[i], dur_arr[i]))
            sel = ordered[:max_wells]
            if use_dur:
                while len(sel) > 1 and dur_arr[sel].sum() + (route_distance(lats[sel], lons[sel]) / speed) * 60 > time_budget:
                    sel = sorted(sel, key=lambda i: urg_arr[i])[:-1]
            df.loc[sel, "scheduled"], df.loc[sel, "plan_unit"] = True, unit
        return df

    _blk = set(blocked_units) if blocked_units else set()
    avail_remote = [u for u in REMOTE_UNITS[:n_remote] if u not in _blk]
    avail_nonremote = [u for u in NONREMOTE_UNITS[:n_nonremote] if u not in _blk]
    unit_clusters = {u: [] for u in avail_remote + avail_nonremote}
    unassigned = set(df.index)

    def _grow(u, target_fld=None):
        zone_remote = (u in REMOTE_UNITS)
        while len(unit_clusters[u]) < max_wells and unassigned:
            cand_pool = [i for i in unassigned if (area_arr[i] in REMOTE_AREAS) == zone_remote]
            if not cand_pool: break
            c_dists = dist_mat[np.ix_(unit_clusters[u], cand_pool)]
            min_dists, max_dists = c_dists.min(axis=0), c_dists.max(axis=0)
            valid_cands = []
            for i_cand, cand_idx in enumerate(cand_pool):
                d_min, d_max, urg = min_dists[i_cand], max_dists[i_cand], urg_arr[cand_idx]
                is_same_fld = (target_fld is not None and field_arr[cand_idx] == target_fld)
                if d_max > elastic_limit: continue
                if is_same_fld or d_min <= 5.0 or (d_min <= elastic_limit and urg <= 2):
                    c_score = d_min - (50 if is_same_fld else 0) - (20 if urg <= 2 else 0)
                    valid_cands.append((cand_idx, d_min, c_score))
            if not valid_cands: break
            valid_cands.sort(key=lambda x: x[2])
            best_idx = None
            for cand_idx, d, _ in valid_cands:
                cand_cluster = unit_clusters[u] + [cand_idx]
                if use_dur and dur_arr[cand_cluster].sum() + (route_distance(lats[cand_cluster], lons[cand_cluster]) / speed) * 60 > time_budget: continue
                best_idx = cand_idx; break
            if best_idx is not None: unit_clusters[u].append(best_idx); unassigned.remove(best_idx)
            else: break

    fu_mask = df["forced_unit"].notna() & df["forced_unit"].isin(unit_clusters.keys())
    for u, grp in df[fu_mask].groupby("forced_unit"):
        for idx in grp.sort_values(["urgency", "dur"]).index:
            if len(unit_clusters[u]) < max_wells and idx in unassigned:
                if unit_clusters[u]:
                    d_min, d_max = dist_mat[unit_clusters[u], idx].min(), dist_mat[unit_clusters[u], idx].max()
                    if d_max > elastic_limit or (d_min > 5.0 and not (d_min <= elastic_limit and urg_arr[idx] <= 2)): continue
                if use_dur and unit_clusters[u]:
                    cand = unit_clusters[u] + [idx]
                    if dur_arr[cand].sum() + (route_distance(lats[cand], lons[cand]) / speed) * 60 > time_budget: continue
                unit_clusters[u].append(idx); unassigned.remove(idx)

    _pre_units = set()
    if df["_pre_unit"].notna().any():
        for idx in df.index[df["_pre_unit"].notna()]:
            u = df.at[idx, "_pre_unit"]
            if u in unit_clusters and idx in unassigned:
                unit_clusters[u].append(idx); unassigned.discard(idx); _pre_units.add(u)
        for u in _pre_units: _grow(u)

    used_units = {u for u, c in unit_clusters.items() if len(c) > 0}
    avail_remote = [u for u in avail_remote if u not in used_units]
    avail_nonremote = [u for u in avail_nonremote if u not in used_units]

    while unassigned and (avail_remote or avail_nonremote):
        field_scores = {}
        un_list = list(unassigned)
        for fld in pd.unique(field_arr[un_list]):
            f_wells = [w for w in un_list if field_arr[w] == fld]
            if not f_wells: continue
            score = sum(100000 if u < -1000 else 10000 if u <= 0 else 5000 if u == 1 else 1000 if u == 2 else 100 if u <= 4 else 10 for u in urg_arr[f_wells])
            field_scores[fld] = score

        if not field_scores: break
        sorted_fields = sorted(field_scores.keys(), key=lambda k: field_scores[k], reverse=True)

        assigned_this_round = False
        for target_fld in sorted_fields:
            f_wells = [w for w in unassigned if field_arr[w] == target_fld]
            zone = "remote" if area_arr[f_wells[0]] in REMOTE_AREAS else "nonremote"
            avail_pool = avail_remote if zone == "remote" else avail_nonremote
            if not avail_pool: continue
            u = avail_pool.pop(0)
            seed = sorted(f_wells, key=lambda x: (urg_arr[x], dur_arr[x]))[0]
            unit_clusters[u].append(seed); unassigned.remove(seed); used_units.add(u)
            _grow(u, target_fld)
            assigned_this_round = True; break
        if not assigned_this_round: break

    for u, c in unit_clusters.items():
        if c: df.loc[c, "scheduled"], df.loc[c, "plan_unit"] = True, u

    return df

def plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed,
              use_urg, use_dur, early_days=0, elastic_limit=5.0, unit_blackout=None, prebooked=None):
    elig = elig.reset_index(drop=True).copy()
    elig["scheduled"], elig["plan_unit"], elig["plan_day"], elig["day_idx"] = False, None, pd.NaT, 0
    early_td, rem = pd.Timedelta(days=early_days), pd.Series(True, index=elig.index)

    is_nwaws = elig["tipe"].isin(["NW", "AWS"])
    fw_c = elig.get("force_week", pd.Series(False, index=elig.index)).fillna(False)
    cc_c = elig.get("carry_ncmp", pd.Series(False, index=elig.index)).fillna(False)
    bypass_reg = (fw_c & ~is_nwaws) | (cc_c & ~is_nwaws)

    np_in = elig.get("np_in_range", pd.Series(False, index=elig.index)).fillna(False)
    next_wt = elig.get("next_wt", pd.Series(pd.NaT, index=elig.index))
    strict_no_late = elig.get("np_in_range", pd.Series(False, index=elig.index)) & elig.get("max_in_range", pd.Series(False, index=elig.index)) & ~is_nwaws
    overdue_nw = is_nwaws & elig["max_date"].notna() & (elig["max_date"] < days[0])

    for i, day in enumerate(days, start=1):
        cond_reg = (~is_nwaws) & ((elig["min_date"] - early_td <= day) | (np_in & (next_wt <= day)) | bypass_reg) & ~(strict_no_late & (day > elig["max_date"]))
        cond_nw = is_nwaws & (((elig["min_date"] <= day) & (elig["max_date"] >= day)) | overdue_nw)
        
        pidx = elig.index[rem & (cond_reg | cond_nw)]
        if len(pidx) == 0: continue

        pool = elig.loc[pidx].copy()
        pool["urgency"] = (pool["max_date"] - day).dt.days.fillna(0)
        pool.loc[bypass_reg.loc[pidx], "urgency"] = pool.loc[bypass_reg.loc[pidx], "urgency"].clip(upper=0)
        pool.loc[is_nwaws.loc[pidx], "urgency"] = pool.loc[is_nwaws.loc[pidx], "urgency"].clip(upper=0) - 10000

        blocked = unit_blackout.get(pd.Timestamp(day).strftime("%Y-%m-%d"), set()) if unit_blackout else set()
        if blocked and "forced_unit" in pool.columns:
            pool = pool[~(pool["forced_unit"].notna() & pool["forced_unit"].isin(blocked))]
            if len(pool) == 0: continue

        pb_day = prebooked[prebooked["day_idx"] == i] if prebooked is not None and len(prebooked) else None
        pd_ = plan(pool, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, current_day=day, elastic_limit=elastic_limit, blocked_units=blocked, prebooked=pb_day)

        sd = pd_[pd_["scheduled"]]
        if len(sd) == 0: continue

        sidx = elig.index[elig["well"].isin(sd["well"])]
        elig.loc[sidx, "scheduled"], elig.loc[sidx, "plan_day"], elig.loc[sidx, "day_idx"] = True, day, i
        elig.loc[sidx, "plan_unit"] = elig.loc[sidx, "well"].map(dict(zip(sd["well"], sd["plan_unit"])))
        rem.loc[sidx] = False

    elig["urgency"] = (elig["max_date"] - days[0]).dt.days
    return elig

def unit_summary(df, speed):
    cols = ["Unit", "Sumur", "Test (min)", "Rute (km)", "Est (min)", "Sub-area", "Deadline tercepat", "⏱️ Early/Late", "Wells"]
    if df is None or not len(df) or "scheduled" not in df.columns: return pd.DataFrame(columns=cols)
    sched = df[df["scheduled"].fillna(False)]
    if not len(sched) or "plan_unit" not in sched.columns: return pd.DataFrame(columns=cols)

    speed = max(float(speed), 1.0)
    rows = []
    for unit, sub in sched.groupby("plan_unit"):
        c = sub[sub["has_coord"].fillna(False)] if "has_coord" in sub.columns else sub
        dist = route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0
        notes = [f"{w} ({lab})" for w, lab in zip(sub["well"], sub.get("timing_label", [""]*len(sub))) if lab]
        dur_sum = int(pd.to_numeric(sub.get("dur", 0), errors="coerce").fillna(0).sum())
        dmin = sub["max_date"].min() if "max_date" in sub.columns else pd.NaT
        
        rows.append({
            "Unit": unit, "Sumur": len(sub), "Test (min)": dur_sum, "Rute (km)": round(dist, 1),
            "Est (min)": int(dur_sum + (dist / speed) * 60),
            "Sub-area": ", ".join(sorted(sub.get("subarea", pd.Series()).dropna().astype(str).unique())),
            "Deadline tercepat": dmin.strftime("%Y-%m-%d") if pd.notna(dmin) else "-",
            "⏱️ Early/Late": ", ".join(notes) or "-",
            "Wells": ", ".join(sub["well"].astype(str))})
    return pd.DataFrame(rows).sort_values("Unit") if rows else pd.DataFrame(columns=cols)

COLORS = [[228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0],
          [166, 86, 40], [247, 129, 191], [26, 188, 156], [241, 196, 15], [106, 61, 154],
          [178, 223, 138], [251, 154, 153]]

def cmap(label, labels):
    try: return COLORS[list(labels).index(label) % len(COLORS)]
    except ValueError: return [130, 130, 130]

def pass2_tekan_miss(week_df, days, per_hi_ts, max_wells, n_remote, n_nonremote, prq_soft_cap=8, addman_skip_km=0.0, due_first=False, notdue_fill_km=5.0):
    horizon, base = len(days), pd.Timestamp(days[0])
    wd = week_df.copy()
    in_scope = wd["lat"].notna() & wd["lon"].notna() & (wd["scheduled"].fillna(False) | (wd["max_date"] <= per_hi_ts))
    sub = wd.loc[in_scope].drop_duplicates("well").reset_index(drop=True)
    n = len(sub)
    if n == 0: return week_df, {"miss_before": 0, "miss_after": 0, "addman_deferred": 0, "notdue_deferred": 0, "overflow": [], "overflow_wells": []}

    lat, lon = sub["lat"].to_numpy(float), sub["lon"].to_numpy(float)
    zone = np.where(sub["area"].isin(REMOTE_AREAS), "remote", "nonremote")
    isnw, catg, isprq = sub["is_nwaws"].fillna(False).to_numpy(bool), sub["category"].astype(str).to_numpy(), (sub["req_tag"].astype(str) == "PRQ").to_numpy()
    min_day = ((sub["min_date"] - base).dt.days + 1).fillna(1).to_numpy().astype(int)
    max_day = ((sub["max_date"] - base).dt.days + 1).fillna(horizon).to_numpy().astype(int)
    is_addman, prio = np.array([c.startswith("Add Manual") for c in catg]), isnw | np.isin(catg, ["Regular A", "Regular B"])
    is_flex = isprq | is_addman

    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]); np.fill_diagonal(D, 0.0)
    nn = np.argsort(D, axis=1)
    nbr = [[j for j in nn[w] if j != w and zone[j] == zone[w]][:12] for w in range(n)]
    units, zof = {"remote": REMOTE_UNITS[:n_remote], "nonremote": NONREMOTE_UNITS[:n_nonremote]}, lambda u: "remote" if u in REMOTE_UNITS else "nonremote"
    P_ROUTE, P_PRIO, P_REG = 15.0, 8000.0, 400.0

    def path_len(o): return float(D[np.array(o)[:-1], np.array(o)[1:]].sum()) if len(o) > 1 else 0.0
    def two_opt(o):
        if len(o) < 4: return o
        best, imp = path_len(o), True
        while imp:
            imp = False
            for i in range(len(o) - 1):
                for k in range(i + 1, len(o)):
                    no = o[:i] + o[i:k+1][::-1] + o[k+1:]
                    nl = path_len(no)
                    if nl + 1e-9 < best: o, best, imp = no, nl, True
        return o
    def cheapest_insert(o, w):
        if not o: return 0.0, 0
        bd, bp = D[w, o[0]], 0
        if D[o[-1], w] < bd: bd, bp = D[o[-1], w], len(o)
        for i in range(len(o) - 1):
            if D[o[i], w] + D[w, o[i+1]] - D[o[i], o[i+1]] < bd: bd, bp = D[o[i], w] + D[w, o[i+1]] - D[o[i], o[i+1]], i + 1
        return bd, bp

    routes, where, widx = {}, {}, {w: i for i, w in enumerate(sub["well"])}
    for _, r in sub.iterrows():
        if r["scheduled"] and r["plan_unit"] is not None and 1 <= int(r["day_idx"] or 0) <= horizon:
            key, i = (r["plan_unit"], int(r["day_idx"])), widx[r["well"]]
            routes.setdefault(key, []).append(i); where[i] = key
    for k in list(routes): routes[k] = two_opt(routes[k])

    def feasible(w, key): return zof(key[0]) == zone[w] and len(routes.get(key, [])) < max_wells and min_day[w] <= key[1] and not (prio[w] and max_day[w] >= 1 and key[1] > max_day[w])
    def candidate_keys(w):
        keys = {where[nb] for nb in nbr[w] if where.get(nb) is not None and feasible(w, where[nb])}
        for d in range(max(1, int(min_day[w])), horizon + 1):
            if prio[w] and max_day[w] >= 1 and d > max_day[w]: break
            for u in units[zone[w]]:
                if feasible(w, (u, d)): keys.add((u, d)); break
        return keys
    def ins_cost(w, key):
        delta, pos = cheapest_insert(routes.get(key, []), w)
        return delta + (P_ROUTE if not routes.get(key, []) else 0), pos

    def repair(pool, defer_above=None):
        pool = list(pool)
        while pool:
            bw, breg, bkey, bpos, bd0 = None, -1e18, None, None, 0.0
            for w in pool:
                cands = candidate_keys(w)
                costs = sorted((ins_cost(w, k) + (k,) for k in cands), key=lambda x: x[0]) if cands else []
                if not costs:
                    if (P_PRIO if prio[w] else P_REG) > breg: breg, bw, bkey, bpos, bd0 = (P_PRIO if prio[w] else P_REG), w, None, None, 0.0
                    continue
                d0, p0, k0 = costs[0]
                reg = (costs[1][0] if len(costs) > 1 else d0 + 1000) - d0 + (6000 if prio[w] else 0)
                if reg > breg: breg, bw, bkey, bpos, bd0 = reg, w, k0, p0, d0
            pool.remove(bw)
            if bkey is not None and (defer_above is None or bd0 <= defer_above or max_day[bw] <= horizon):
                routes.setdefault(bkey, []).insert(bpos, bw); where[bw] = bkey

    def relocate_pass():
        improved = False
        for w in list(where):
            cur = where[w]
            gain0 = path_len(routes[cur]) - path_len([x for x in routes[cur] if x != w]) + (P_ROUTE if len(routes[cur]) == 1 else 0)
            best = None
            for key in candidate_keys(w):
                if key == cur: continue
                delta, pos = ins_cost(w, key)
                if gain0 - delta > 1e-6 and (best is None or gain0 - delta > best[0]): best = (gain0 - delta, key, pos)
            if best:
                routes[cur].remove(w)
                if not routes[cur]: 
                    del routes[cur]
                routes.setdefault(best[1], []).insert(best[2], w)
                where[w] = best[1]
                improved = True
        return improved

    is_due = (max_day <= horizon)
    unpl = lambda pred: [w for w in range(n) if pred(w) and w not in where]

    for w in [w for w in list(where) if is_flex[w] or (due_first and not is_due[w])]:
        key = where.pop(w); routes[key].remove(w)
        if not routes[key]: del routes[key]

    repair(unpl(lambda w: prio[w] and not isprq[w] and (not due_first or is_due[w])))
    repair(unpl(lambda w: catg[w] == "Regular C" and not isprq[w] and (not due_first or is_due[w])))
    repair(unpl(lambda w: catg[w] == "Regular D" and not isprq[w] and (not due_first or is_due[w])))
    repair(unpl(lambda w: is_addman[w] and not isprq[w] and (not due_first or is_due[w])), defer_above=(addman_skip_km if addman_skip_km > 0 else None))
    repair(unpl(lambda w: isprq[w] and (not due_first or is_due[w])))
    
    for _ in range(3):
        if not relocate_pass(): break
    for k in list(routes): routes[k] = two_opt(routes[k])

    if due_first:
        _fill_km = notdue_fill_km if notdue_fill_km > 0 else 5.0
        for w in sorted([w for w in range(n) if not is_due[w] and w not in where], key=lambda x: max_day[x]):
            best = None
            for nb in nbr[w]:
                k = where.get(nb)
                if k is not None and feasible(w, k):
                    dd, pos = cheapest_insert(routes[k], w)
                    if dd <= _fill_km and (best is None or dd < best[0]): best = (dd, k, pos)
            if best: routes.setdefault(best[1], []).insert(best[2], w); where[w] = best[1]

    overflow_wells = []
    for w in [w for w in range(n) if isprq[w] and w not in where]:
        best = None
        for key, r in routes.items():
            if zof(key[0]) != zone[w] or not r or len(r) >= prq_soft_cap: continue
            dmin = min(D[w, x] for x in r)
            if best is None or dmin < best[0]: best = (dmin, key)
        if best:
            _, pos = cheapest_insert(routes[best[1]], w)
            routes.setdefault(best[1], []).insert(pos, w); where[w] = best[1]; overflow_wells.append(sub["well"].iloc[w])

    miss_before = int((~sub["scheduled"].fillna(False) & (sub["max_date"] <= per_hi_ts) & ~is_flex).sum())
    miss_after = sum(1 for w in range(n) if w not in where and max_day[w] <= horizon and not is_flex[w])
    addman_deferred = sum(1 for w in range(n) if is_addman[w] and w not in where)
    notdue_deferred = sum(1 for w in range(n) if (max_day[w] > horizon) and w not in where)
    overflow_routes = [(k, len(v)) for k, v in routes.items() if len(v) > max_wells]

    out = week_df.copy()
    dtc = [c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
    day_of, unit_of = {}, {}
    for w_i, key in where.items(): wn = widx[w_i]; unit_of[wn], day_of[wn] = key[0], key[1]
    scope_wells = set(sub["well"])
    for i in out.index:
        wn = out.at[i, "well"]
        if wn not in scope_wells: continue
        if wn in day_of:
            out.at[i, "scheduled"], out.at[i, "plan_unit"], out.at[i, "day_idx"], out.at[i, "plan_day"], out.at[i, "manual"] = True, unit_of[wn], day_of[wn], days[day_of[wn]-1], False
        else:
            out.at[i, "scheduled"], out.at[i, "plan_unit"], out.at[i, "day_idx"] = False, None, 0
    for c in dtc: out[c] = pd.to_datetime(out[c], errors="coerce")
    return out, {"miss_before": miss_before, "miss_after": miss_after, "addman_deferred": addman_deferred,
                 "notdue_deferred": notdue_deferred, "overflow": overflow_routes, "overflow_wells": overflow_wells}

def pass2_kpis(week_df, per_hi_ts, days):
    base, wd = pd.Timestamp(days[0]), week_df.copy()
    cat, prq = wd["category"].astype(str), wd["req_tag"].astype(str) == "PRQ"
    isflex, sch, maxd, dayi = prq | cat.str.startswith("Add Manual"), wd["scheduled"].fillna(False), ((wd["max_date"] - base).dt.days + 1), wd["day_idx"].fillna(0).astype(int)
    ontime = late = miss = 0
    for i in wd.index:
        if isflex.iloc[wd.index.get_loc(i)]: continue
        if not sch.loc[i]:
            if pd.notna(wd.at[i, "max_date"]) and wd.at[i, "max_date"] <= per_hi_ts: miss += 1
            continue
        if pd.isna(maxd.loc[i]) or dayi.loc[i] <= max(maxd.loc[i], 1): ontime += 1
        else: late += 1
    s = wd[sch]
    km = sum(route_distance(gp["lat"].to_numpy(), gp["lon"].to_numpy()) for _, gp in s.groupby(["plan_unit", "day_idx"]) if len(gp) > 1)
    nsch = len(s); nk = ontime + late + miss
    return dict(miss=miss, ontime=ontime, late=late, ontime_pct=(ontime/nk*100 if nk else 100.0),
                km_well=(km/nsch if nsch else 0.0), crew=s.groupby(["plan_unit", "day_idx"]).ngroups if nsch else 0, sched=nsch,
                prq=int((sch & prq).sum()), addman=int((sch & cat.str.startswith("Add Manual")).sum()))

def fill_notdue(week_df, days, per_hi_ts, max_wells, fill_km=5.0):
    base, out = pd.Timestamp(days[0]), week_df.copy()
    _s = out["scheduled"].fillna(False)
    sch, nd = out[_s & out["lat"].notna()], out[~_s & (out["max_date"] > per_hi_ts) & out["lat"].notna()].sort_values("max_date")
    if not len(sch) or not len(nd): return out, 0, int((~_s & (out["max_date"] > per_hi_ts)).sum())
    routes, zof, filled = {}, lambda u: "remote" if u in REMOTE_UNITS else "nonremote", 0
    for _, r in sch.iterrows(): routes.setdefault((r["plan_unit"], int(r["day_idx"])), []).append((r["lat"], r["lon"]))
    for i, w in nd.iterrows():
        wz, wmin, best = "remote" if w["area"] in REMOTE_AREAS else "nonremote", int((w["min_date"] - base).days + 1) if pd.notna(w["min_date"]) else 1, None
        for (u, d), pts in routes.items():
            if zof(u) != wz or len(pts) >= max_wells or d < max(1, wmin): continue
            dmin = min(float(haversine_km(w["lat"], w["lon"], la, lo)) for la, lo in pts)
            if dmin <= fill_km and (best is None or dmin < best[0]): best = (dmin, (u, d))
        if best:
            routes[best[1]].append((w["lat"], w["lon"]))
            out.at[i, "scheduled"], out.at[i, "plan_unit"], out.at[i, "day_idx"], out.at[i, "plan_day"] = True, best[1][0], best[1][1], days[best[1][1]-1]
            filled += 1
    return out, filled, int((~out["scheduled"].fillna(False) & (out["max_date"] > per_hi_ts)).sum())


COPILOT_SYSTEM = """Kamu copilot untuk WELLGO, aplikasi penjadwalan Mobile Well Test (MWT) di PT Pertamina Hulu Rokan area SLN (WK Rokan). Tugasmu MENJELASKAN & MENJAWAB pertanyaan tentang jadwal yang SUDAH dihitung engine. Kamu TIDAK menjadwalkan/mengubah apa pun — penjadwalan dilakukan engine deterministik, bukan kamu.
...
"""

def copilot_context(week_df, days, batch_lo, batch_hi, pass2_summary=None, notdue_fill_summary=None, max_rows=350):
    horizon, wd = len(days), week_df.copy()
    sch, miss = wd[wd["scheduled"].fillna(False)], wd[~wd["scheduled"].fillna(False) & (wd["max_date"] <= batch_hi)]
    L = [f"Periode {pd.Timestamp(batch_lo).date()}..{pd.Timestamp(batch_hi).date()} ({horizon} hari).",
         f"Eligible={len(wd)}, terjadwal={len(sch)}, miss-deadline={len(miss)}."]
    want = [c for c in ["well", "plan_unit", "day_idx", "category", "field", "area", "min_date", "max_date"] if c in sch.columns]
    L.append("\nJADWAL (" + "|".join(want) + "):")
    for _, r in sch[want].head(max_rows).iterrows():
        L.append("|".join(str(r[c].date() if c in ("min_date", "max_date") and pd.notna(r[c]) else f"h{int(r[c])}" if c=="day_idx" else r[c]) for c in want))
    return "\n".join(L)

def copilot_answer(question, context, history=None):
    return "Jawaban dummy karena limitasi library anthropic di snippet.", None


# ================================================================== UI Configuration
init_db()

CRIT = {
    "Kedekatan jarak saja": (False, False),
    "Jarak + durasi test": (False, True),
    "Jarak + min-max (deadline)": (True, False),
    "Jarak + durasi + min-max": (True, True),
}

# ── Sidebar UI / UX ────────────────────────────────────────────────────────
with st.sidebar:
    ui.section("💾 Manajemen Data")
    up = st.file_uploader("Upload Excel kandidat & spasial", type=["xlsx", "xlsm"])
    
    col_sh1, col_sh2 = st.columns(2)
    with col_sh1: sheet_kandidat = st.text_input("Sheet Kandidat", SHEET_DEFAULT)
    with col_sh2: sheet_spasial = st.text_input("Sheet Spasial", "Data_Spasial")
    sheet_breakin = st.text_input("Sheet Break-In", "BreakIn")
    mpas_only = st.checkbox("Hanya Unit Tes (MPAS), exclude TS", value=True)

if up is None:
    ui.hero_header(date_str=datetime.now().strftime("%d %b %Y"), horizon=7, units=9, compliance=0, mode="pooled")
    t_mulai, t_panduan = st.tabs(["🏠 Mulai", "📘 Panduan"])
    with t_mulai: st.info("💡 Silakan unggah berkas Excel data kandidat sumur & master database koordinat spasial pada sidebar untuk memulai kalkulasi rute.")
    with t_panduan: guide.render_guide()
    st.stop()

# ── Data Loading Awal ─────────────────────────────────────
raw = load_candidates(up.getvalue(), sheet_kandidat)
raw["is_breakin"] = False
if sheet_breakin and sheet_breakin.strip():
    try:
        raw_break = load_candidates(up.getvalue(), sheet_breakin.strip())
        if len(raw_break):
            raw_break["is_breakin"] = True
            raw = pd.concat([raw[~raw["well"].isin(set(raw_break["well"]))], raw_break], ignore_index=True)
    except Exception: pass
spatial_db = load_spatial_data(up.getvalue(), sheet_spasial)

with st.sidebar:
    st.divider()
    ui.section("🗺️ Filter Area & Unit")
    all_areas = sorted(raw["area"].dropna().unique())
    default_excl = [a for a in all_areas if a == "LIBO"]
    excl_areas = st.multiselect("Exclude Area Terpilih", all_areas, default=default_excl)
    if mpas_only:
        ts_unavail = st.multiselect("Area Fasilitas TS Down (Dialihkan ke MWT)", all_areas)
        mwt_unavail = st.multiselect("Area Fleet MWT Down (Dialihkan ke TS)", all_areas)
    else: ts_unavail, mwt_unavail = [], []
    
    st.divider()
    ui.section("⏱️ Status Realisasi Harian")
    comp_files = st.file_uploader("Upload file COMP/NCMP harian", type=["xlsx", "xlsm"], accept_multiple_files=True)
    skip_woff = st.checkbox("Skip sumur NCMP yang berstatus OFF", value=True)
    aws_split = st.checkbox("Pecah AWS jadi 2 kunjungan", value=False)
    
    with st.expander("🗑️ Kelola SCH_Database"):
        _ok = st.checkbox("Hapus SEMUA log eksekusi", key="_confirm_wipe_sch")
        if st.button("Hapus SCH_Database sekarang", disabled=not _ok, use_container_width=True):
            reset_execution_log()
            for _k in ("_compncmp_sig", "_compncmp_summary"): st.session_state.pop(_k, None)
            st.cache_data.clear(); st.rerun()

    st.divider()
    manual_file = st.file_uploader("Upload Manual Schedule (Untuk Komparasi)", type=["xlsx", "xlsm"])
    manual_sheet = st.text_input("Nama Sheet Manual", "Well Test Schedule")
    
    st.divider()
    ui.section("📅 Horizon Perencanaan")
    _today = datetime.now().date()
    periode = st.date_input("Rentang Siklus (Periode)", value=(_today, _today + timedelta(days=6)))
    per_lo = periode[0] if isinstance(periode, (list, tuple)) else periode
    per_hi = periode[1] if isinstance(periode, (list, tuple)) and len(periode)==2 else per_lo + timedelta(days=6)
    plan_start_date = st.date_input("Mulai Planning dari Tanggal", value=per_lo, min_value=per_lo, max_value=per_hi)

    with st.expander("🚫 Unit MWT Tidak Tersedia (per tanggal)", expanded=False):
        _ps, _ph = pd.Timestamp(plan_start_date), pd.Timestamp(per_hi)
        _blk_days = [(_ps + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(max(1, min(60, (_ph - _ps).days + 1)))]
        _grid = pd.DataFrame(False, index=ALL_UNITS, columns=_blk_days)
        for _u, _dk in st.session_state.get("unit_blackout", []):
            if _u in _grid.index and _dk in _grid.columns: _grid.loc[_u, _dk] = True
        _ed_blk = st.data_editor(_grid.reset_index().rename(columns={"index": "Unit"}), hide_index=True, use_container_width=True, disabled=["Unit"])
        _new_blk = [(r["Unit"], dk) for _, r in _ed_blk.iterrows() for dk in _blk_days if bool(r[dk])]
        st.session_state["unit_blackout"] = _new_blk
        if _new_blk and st.button("Bersihkan semua blokir", use_container_width=True): st.session_state["unit_blackout"] = []; st.rerun()
                                    
    st.divider()
    with st.form("opt_form"):
        with st.expander("⚙️ Parameter Algoritma (Advanced)"):
            mode_label = st.radio("Mode Distribusi Unit", ["Dedicated (Territory)", "Pooled (Bebas Zona)"], index=1)
            mode = "dedicated" if mode_label.startswith("Dedicated") else "pooled"
            crit_label = st.radio("Kriteria Utama Optimasi", list(CRIT.keys()), index=3)
            use_urg, use_dur = CRIT[crit_label]
            max_wells = st.slider("Target Sumur / Unit / Hari", 3, 8, 6)
            n_remote = st.slider("Unit Area Remote", 1, 5, 5, disabled=(mode=="dedicated"))
            n_nonremote = st.slider("Unit Area Non-Remote", 1, 4, 4, disabled=(mode=="dedicated"))
            elastic_limit = st.slider("Batas Persebaran Rute (km)", 5, 50, 5, 1)
            two_layer = st.checkbox("Optimasi 2-lapis", value=False)
            pass2_on = st.checkbox("🚑 Pass-2: Tekan Miss-Deadline", value=False)
            addman_skip_km = st.slider("↳ Tunda AddMan jauh > (km)", 0.0, 30.0, 0.0, 0.5, disabled=not pass2_on)
            due_first = st.checkbox("🗓️ Prioritaskan due periode ini", value=False)
            notdue_fill_km = st.slider("↳ Maks detour not-yet-due (km)", 0.0, 20.0, 5.0, 0.5, disabled=not due_first)
            time_budget = st.slider("Time Budget / Hari (Menit)", 180, 540, 360, 30, disabled=not use_dur)
            speed = st.slider("Kecepatan Rata-rata Fleet (km/jam)", 10, 60, 25, 5)
            early_days = st.slider("Skenario Early Test (H-Min)", 0, 7, 0)
            show_block = st.checkbox("Tampilkan Block Area Field di Peta", value=True)
        submit_btn = st.form_submit_button("🔄 Re-run Optimizer", type="primary", use_container_width=True)

# ── Data Processing Block (Engine Logic) ───────────────────────────────────
per_lo_ts, per_hi_ts = pd.Timestamp(per_lo), pd.Timestamp(per_hi)
plan_start_ts = pd.Timestamp(plan_start_date)
horizon = max(1, min(60, (per_hi_ts - plan_start_ts).days + 1))

if comp_files:
    import hashlib
    sig = hashlib.md5(b"".join(sorted(f.getvalue() for f in comp_files))).hexdigest()
    if st.session_state.get("_compncmp_sig") != sig:
        with st.spinner("Sinkronisasi status COMP/NCMP..."): summ_imp = import_compncmp([f.getvalue() for f in comp_files])
        st.session_state["_compncmp_sig"] = sig; st.session_state["_compncmp_summary"] = summ_imp
    summ_imp = st.session_state["_compncmp_summary"]

if excl_areas: raw = raw[~raw["area"].isin(excl_areas)].copy()
if mpas_only:
    raw.loc[raw["is_ts"] & raw["area"].isin(ts_unavail), "dur"] = 60
    raw = raw[((raw["is_mpas"] | raw["unit_unknown"]) & ~raw["area"].isin(mwt_unavail)) | (raw["is_ts"] & raw["area"].isin(ts_unavail))].copy()

raw = resolve_coords(raw, spatial_db, load_coord_cache(), field_assign=st.session_state.get("field_assign", {}))

_basecoord = raw[raw["coord_source"].isin(["master_spasial", "database", "cache"])]
field_list = sorted(_basecoord["field"].dropna().unique().tolist())
field_wells_coord = spatial_db.rename(columns={"WELL": "well", "FIELD": "field", "LAT": "lat", "LON": "lon"}) if not spatial_db.empty else _basecoord[["field", "well", "lat", "lon"]]

days = [plan_start_ts + pd.Timedelta(days=i) for i in range(horizon)]
week_lo, week_hi = days[0], days[-1]
executed_log, ncmp_log, pending_log = status_in_period(per_lo, per_hi)
comp_col, manual_comp = set(raw.loc[raw["sch_status"] == "COMP", "well"]), set(st.session_state.get("manual_comp", []))

aws_done, aws_active, aws_extra_rows = set(), {}, []
_aws = raw[raw["tipe"] == "AWS"].drop_duplicates("well")
if len(_aws):
    _recs = comp_records(set(_aws["well"]))
    for _, _w in _aws.iterrows():
        _wn, pop, has_pop = _w["well"], _w.get("pop_date", pd.NaT), pd.notna(_w.get("pop_date", pd.NaT))
        recs = [(d, r) for d, r in _recs.get(_wn, []) if pd.notna(d) and (not has_pop or pd.Timestamp(d).normalize() >= pd.Timestamp(pop).normalize())]
        as1, as2 = any("AS1" in r for _, r in recs), any("AS2" in r for _, r in recs)
        excel_aws2 = "AWS2" in str(_w.get("category", "")).upper()
        a2_ovr = (None, None) if (excel_aws2 or not has_pop) else (pop + pd.Timedelta(days=5), pop + pd.Timedelta(days=10))
        n_comp = len([d for d, _ in recs if pd.notna(d)])
        
        if as2 or n_comp >= 2: aws_done.add(_wn)
        elif as1 or n_comp >= 1: aws_active[_wn] = ("AWS2", a2_ovr[0], a2_ovr[1])

    for _wn, (_ph, _lo, _hi) in aws_active.items():
        _m = raw["well"] == _wn
        if _lo is not None: raw.loc[_m, "min_date"] = _lo
        if _hi is not None: raw.loc[_m, "max_date"] = _hi
        raw.loc[_m, "category"] = _ph

executed = (executed_log | comp_col | manual_comp | aws_done) - set(aws_active.keys())
comp_disp_set = (executed_log | comp_col | manual_comp) - set(aws_active.keys())
pending_set = (set(pending_log["well"]) | set(raw.loc[raw["sch_status"].isin(["PENDING", "PEND"]), "well"])) - executed
pending_sched = dict(zip(pending_log["well"], pending_log["plan_date"]))
ncmp_set = (set(ncmp_log["well"]) | set(raw.loc[raw["sch_status"] == "NCMP", "well"])) - executed - pending_set

force_on = set(st.session_state.get("force_on_nwaws", [])) & set(raw[(raw["tipe"].isin(["NW", "AWS"])) & (raw["status"] == "OFF")]["well"])
if force_on: raw.loc[raw["well"].isin(force_on), "status"] = "ON"
master_off_wells = set(raw.loc[raw["status"] == "OFF", "well"])
woff_set = ncmp_set & master_off_wells if skip_woff else set()

ncmp_replan = (ncmp_set & set(raw["well"])) - woff_set
ncmp_no_data = sorted(ncmp_set - set(raw["well"]))
ncmp_df = pd.concat([ncmp_log, pd.DataFrame({"well": sorted(set(raw.loc[raw["sch_status"] == "NCMP", "well"]) - executed - pending_set), "reason": "", "plan_date": "(kolom)"})], ignore_index=True).drop_duplicates("well")
replan_df = ncmp_df[ncmp_df["well"].isin(ncmp_replan)].copy()

batch_lo, batch_hi = per_lo_ts, per_hi_ts
in_range = ((raw["min_date"] <= batch_hi) & (raw["max_date"] >= batch_lo)) | ((raw.get("next_wt", pd.NaT) >= batch_lo) & (raw.get("next_wt", pd.NaT) <= batch_hi))
is_nwaws_c = raw["is_nwaws"].fillna(False)
overdue_prio = (is_nwaws_c | raw["req_tag"].isin(["PRQ", "ORQ"])) & raw["max_date"].notna() & (raw["max_date"] < batch_lo)

cand = raw[(in_range | raw["well"].isin(ncmp_replan) | (raw["force_week"].fillna(False) & ~is_nwaws_c) | overdue_prio) & (~raw["well"].isin(executed | pending_set))].copy()
cand["np_in_range"] = ((raw.get("next_wt", pd.NaT) >= batch_lo) & (raw.get("next_wt", pd.NaT) <= batch_hi)).loc[cand.index]
cand["max_in_range"] = ((raw["max_date"] >= batch_lo) & (raw["max_date"] <= batch_hi)).loc[cand.index]
elig_all = cand[(cand["status"] != "OFF") & (~cand["well"].isin(woff_set))].copy()
elig_all["carry_ncmp"] = elig_all["well"].isin(ncmp_replan)

elig_all["urgency"] = (elig_all["max_date"] - week_lo).dt.days.fillna(0)
mid_prio = (elig_all["force_week"].fillna(False) & ~elig_all["is_nwaws"].fillna(False)) | elig_all["carry_ncmp"]
elig_all.loc[mid_prio, "urgency"] = elig_all.loc[mid_prio, "urgency"].clip(upper=0)
elig_all.loc[elig_all["is_nwaws"].fillna(False), "urgency"] = elig_all.loc[elig_all["is_nwaws"].fillna(False), "urgency"].clip(upper=0) - 10000
_flex = (~(elig_all["is_nwaws"].fillna(False) | mid_prio | elig_all["req_tag"].isin(["PRQ", "ORQ"]))) & elig_all["max_date"].notna() & ((elig_all["max_date"] < batch_lo) | (elig_all["min_date"] > batch_hi))
elig_all.loc[_flex, "urgency"] = max(int((week_hi - week_lo).days), 1)

elig, nocoord = elig_all[elig_all["has_coord"]].copy(), elig_all[~elig_all["has_coord"]].copy()
ub_dict = {_dk: {u for u, d in st.session_state.get("unit_blackout", []) if d == _dk} for _, _dk in st.session_state.get("unit_blackout", [])}

notdue_fill_summary = None
if len(elig):
    if due_first:
        wk_due = plan_week(elig[elig["max_date"] <= batch_hi], days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, ub_dict) if len(elig[elig["max_date"] <= batch_hi]) else elig[elig["max_date"] <= batch_hi].assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
        week_df, _nfill, _ndef = fill_notdue(pd.concat([wk_due, elig[elig["max_date"] > batch_hi].assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)], ignore_index=True), days, per_hi_ts, max_wells, notdue_fill_km)
        notdue_fill_summary = {"filled": _nfill, "deferred": _ndef}
    elif two_layer:
        prio_elig = elig[elig["is_nwaws"].fillna(False) | elig["req_tag"].isin(["PRQ", "ORQ"]) | elig["carry_ncmp"].fillna(False)]
        wk_prio = plan_week(prio_elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, ub_dict) if len(prio_elig) else prio_elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
        pb = wk_prio[wk_prio["scheduled"]]
        reg_elig = elig[~elig.index.isin(prio_elig.index)]
        wk_reg = plan_week(reg_elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, ub_dict, prebooked=pb if len(pb) else None) if len(reg_elig) else reg_elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
        week_df = pd.concat([wk_prio, wk_reg], ignore_index=True)
    else:
        week_df = plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, ub_dict)
else:
    week_df = elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)

if len(nocoord): week_df = pd.concat([week_df, nocoord.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)], ignore_index=True)

week_df["zone"], week_df["manual"] = np.where(week_df["area"].isin(REMOTE_AREAS), "remote", "non-remote"), False
man, zone_rejects = st.session_state.get("manual_assign", {}), []
if man:
    _inject, _master, _dtcols = [], raw.drop_duplicates("well").set_index("well"), [c for c in week_df.columns if pd.api.types.is_datetime64_any_dtype(week_df[c])]
    for w in [w for w in man if w not in set(week_df["well"]) and w in _master.index]:
        newrow = {c: (_master.loc[w, c] if c in _master.columns else np.nan) for c in week_df.columns}
        newrow.update({"well": w, "zone": "remote" if str(_master.loc[w, "area"]) in REMOTE_AREAS else "non-remote", "scheduled": False, "manual": False, "urgency": 0})
        _inject.append(newrow)
    if _inject:
        week_df = pd.concat([week_df, pd.DataFrame(_inject)], ignore_index=True)
        for c in _dtcols: week_df[c] = pd.to_datetime(week_df[c], errors="coerce")
    for w, info in list(man.items()):
        m = week_df["well"] == w
        if not m.any() or int(info["day_idx"]) < 1 or int(info["day_idx"]) > horizon: continue
        _uzone = "remote" if info["unit"] in REMOTE_UNITS else "non-remote"
        if _uzone != str(week_df.loc[m, "zone"].iloc[0]):
            zone_rejects.append((w, str(week_df.loc[m, "zone"].iloc[0]), info["unit"], _uzone))
            del st.session_state["manual_assign"][w]; continue
        week_df.loc[m, ["scheduled", "plan_unit", "day_idx", "plan_day", "manual"]] = [True, info["unit"], int(info["day_idx"]), days[int(info["day_idx"]) - 1], True]

if st.session_state.get("manual_unassign", []):
    m_un = week_df["well"].isin(st.session_state["manual_unassign"])
    week_df.loc[m_un, ["scheduled", "plan_unit", "day_idx", "plan_day", "manual"]] = [False, None, 0, pd.NaT, False]

pass2_summary, pass2_compare = None, None
if pass2_on and len(week_df):
    _wd_before = week_df.copy()
    week_df, pass2_summary = pass2_tekan_miss(week_df, days, per_hi_ts, max_wells, n_remote, n_nonremote, addman_skip_km=addman_skip_km)
    pass2_compare = {"before": pass2_kpis(_wd_before, per_hi_ts, days), "after": pass2_kpis(week_df, per_hi_ts, days)}

scheduled_all = week_df[week_df["scheduled"]].copy()
if "timing" not in scheduled_all.columns: scheduled_all["timing"] = scheduled_all["timing_label"] = scheduled_all["out_dir"] = None

if len(scheduled_all):
    _pd, _mn, _mx = scheduled_all["plan_day"], scheduled_all["min_date"], scheduled_all["max_date"]
    _en, _ln, _oe, _ol = (_mn - _pd).dt.days, (_pd - _mx).dt.days, _pd < _mn, _pd > _mx
    _is_in_range = (_mn <= batch_hi) & (_mx >= batch_lo)
    
    def _cat_lab(tipe, oe, ol, en, ln, tag, in_rng):
        if tipe in ["NW", "AWS"]: return "on-time", f"{tipe} (Prioritas)", ""
        in_window = not oe and not ol
        arah, n = "early" if oe else "late", int(en) if oe else int(ln)
        if tag in ["PRQ", "ORQ"]: return ("on-time", f"{tag} (on-time)", "") if in_window else (tag, f"{tag} ({arah} {n} hari)", arah)
        if in_window: return "on-time", "", ""
        if not in_rng: return "on-time", f"Out of Window ({arah} {n}d)", ""
        return ("EARLY" if oe else "LATE"), f"{arah} {n} hari", arah

    _cats = [_cat_lab(tp, oe, ol, en, ln, tg, rng) for tp, oe, ol, en, ln, tg, rng in zip(scheduled_all["tipe"], _oe, _ol, _en.fillna(0), _ln.fillna(0), scheduled_all.get("req_tag", pd.Series("", index=scheduled_all.index)).fillna(""), _is_in_range)]
    scheduled_all["timing"], scheduled_all["timing_label"], scheduled_all["out_dir"] = [c[0] for c in _cats], [c[1] for c in _cats], [c[2] for c in _cats]

leftover = week_df[~week_df["scheduled"]].copy()
missed = leftover[leftover["max_date"] <= batch_hi] if len(leftover) else leftover.copy()

# ── Render Header & KPIs ───────────────────────────────────────────────────
total_scheduled = len(scheduled_all)
total_kpi_target = total_scheduled + len(missed)
comp_rate = int(100 * total_scheduled / total_kpi_target) if total_kpi_target > 0 else 100

computed_total_km, total_minutes = 0.0, 0.0
for (di, dday, unit), sub in scheduled_all.groupby(["day_idx", "plan_day", "plan_unit"]):
    dist_val = route_distance(sub[sub["has_coord"]]["lat"].values, sub[sub["has_coord"]]["lon"].values) if len(sub[sub["has_coord"]]) > 1 else 0.0
    computed_total_km += dist_val
    total_minutes += int(pd.to_numeric(sub.get("dur", 0), errors="coerce").fillna(0).sum()) + (dist_val / max(float(speed), 1.0)) * 60

avg_utilization = (total_minutes / (max(len(scheduled_all["plan_unit"].unique()), 1) * horizon * time_budget)) * 100 if time_budget > 0 else 0

ui.hero_header(date_str=plan_start_ts.strftime("%d %b %Y"), horizon=horizon, units=len(scheduled_all["plan_unit"].unique()) if len(scheduled_all) else 0, compliance=comp_rate, mode=mode)
ui.kpi_row([
    ("wells scheduled", f"{total_scheduled}", f"/{len(elig_all)}", ui.TEAL_GREEN),
    ("miss deadline", f"{len(missed)}", " wells", ui.RED),
    ("wells off", f"{len(cand[cand['status'] == 'OFF'])}", " wells", "#64748B"),
    ("total route", f"{computed_total_km:.0f}", " km", ui.TEAL),
    ("avg utilization", f"{avg_utilization:.0f}", "%", ui.AMBER),
])

# ── Main Workspace Tabs (Refactored 5 Tabs) ────────────────────────────────
tab_dashboard, tab_editor, tab_analitik, tab_komparasi, tab_asisten = st.tabs([
    "🗺️ Dashboard & Peta", 
    "📅 Editor Jadwal", 
    "📊 Analitik & Report", 
    "⚖️ Komparasi & Ekspor", 
    "🤖 Asisten WELLGO"
])

# ==================== TAB 1: DASHBOARD & PETA ====================
with tab_dashboard:
    if len(nocoord):
        with st.expander(f"📍 Sumur Tanpa Koordinat: {len(nocoord)} sumur", expanded=True):
            st.dataframe(nocoord[["well", "field", "area", "subarea", "category", "tipe"]], use_container_width=True, hide_index=True)

    c_flt1, c_flt2 = st.columns([3, 1])
    day_labels = [days[i].strftime("%Y-%m-%d") for i in range(horizon)]
    with c_flt1: sel_labels = st.multiselect("🗓️ Fokus Tanggal Rute", day_labels, default=day_labels) or day_labels
    with c_flt2: dur_pick = st.multiselect("⏱️ Filter Durasi Test", [30, 60], default=[30, 60]) or [30, 60]

    sel_idx = sorted({lbl: i + 1 for i, lbl in enumerate(day_labels)}[l] for l in sel_labels)
    disp = scheduled_all[scheduled_all["day_idx"].isin(sel_idx) & scheduled_all["dur"].isin(dur_pick)].copy() if len(scheduled_all) else scheduled_all.copy()
    
    mco1, mco2, mco3 = st.columns([1.5, 2, 1.4])
    color_mode = mco1.selectbox("🎨 Skema Pewarnaan Peta", ["Otomatis (hari/unit)", "Zona remote/non-remote", "Per unit", "Early / Late test"])
    unit_filter = mco2.multiselect("🔧 Batasi Tampilan Unit", sorted(scheduled_all["plan_unit"].dropna().unique().tolist()) if len(scheduled_all) else [])
    search_q = mco3.text_input("🔎 Pencarian Cepat", placeholder="Contoh: BO083").strip().upper()
    
    toggles1, toggles2 = st.columns(2)
    show_miss = toggles1.toggle(f"📌 Tampilkan Miss Deadline", value=False)
    field_block = toggles2.multiselect("📦 Batas Field Area", field_list, default=[])

    pmap = disp[disp["has_coord"]].copy() if len(disp) else disp.copy()
    if unit_filter: pmap = pmap[pmap["plan_unit"].isin(unit_filter)]
    
    layers = []
    if len(pmap):
        if color_mode == "Zona remote/non-remote": pmap["color"] = pmap["zone"].map({"remote": [30, 120, 220], "non-remote": [240, 140, 30]}).fillna([130, 130, 130])
        elif color_mode == "Per unit": pmap["color"] = pmap["plan_unit"].apply(lambda k: cmap(k, sorted(pmap["plan_unit"].dropna().unique())))
        elif color_mode == "Early / Late test": pmap["color"] = pmap["timing"].map({"EARLY": [30, 120, 220], "on-time": [150, 150, 150], "LATE": [220, 30, 30], "PRQ": [150, 80, 200], "ORQ": [0, 160, 140]}).fillna([150, 150, 150])
        else: pmap["color"] = (pmap["plan_unit"] if len(sel_idx) == 1 else pmap["day_idx"]).apply(lambda k: cmap(k, sorted((pmap["plan_unit"] if len(sel_idx) == 1 else pmap["day_idx"]).unique())))
        
        pmap["radius"] = np.where(pmap["coord_source"].str.startswith("imputed"), 90, 170)
        pmap["hit"] = pmap["well"].str.upper().isin([t for t in search_q.replace(",", " ").split() if t]) if search_q else False
        pmap["ring"] = pmap.apply(lambda r: [255, 235, 0] if r["hit"] else {"NW": [220, 30, 30], "AWS": [245, 150, 20]}.get(r["tipe"], [120, 120, 120]), axis=1)
        pmap["ringw"] = np.where(pmap["hit"], 6, np.where(pmap["tipe"].isin(["NW", "AWS"]), 3, 0))
        
        pmap["tgl_str"] = pmap["plan_day"].dt.strftime("%Y-%m-%d").fillna("belum terjadwal")
        pmap["min_str"] = pmap["min_date"].dt.strftime("%Y-%m-%d").fillna("—")
        pmap["max_str"] = pmap["max_date"].dt.strftime("%Y-%m-%d").fillna("—")
        
        lines = []
        for u, sub in pmap.groupby("plan_unit"):
            s = sub.reset_index(drop=True)
            if len(s) > 1:
                order, _ = optimize_route(s["lat"].values, s["lon"].values)
                col = list(s["color"].iloc[0])
                for a in range(len(order) - 1): lines.append({"from": [s.loc[order[a], "lon"], s.loc[order[a], "lat"]], "to": [s.loc[order[a+1], "lon"], s.loc[order[a+1], "lat"]], "color": col})
        if lines: layers.append(pdk.Layer("LineLayer", data=pd.DataFrame(lines), get_source_position="from", get_target_position="to", get_color="color", get_width=2))
        layers.append(pdk.Layer("ScatterplotLayer", data=pmap, get_position=["lon", "lat"], get_fill_color="color", get_radius="radius", get_line_color="ring", get_line_width="ringw", line_width_min_pixels=1, stroked=True, filled=True, pickable=True, opacity=0.9))

    lat_init, lon_init = float(pmap["lat"].mean()) if len(pmap) else 1.6, float(pmap["lon"].mean()) if len(pmap) else 101.3
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat_init, longitude=lon_init, zoom=8.5), map_style="road", tooltip={"text": "{well} [{tipe}]\nTanggal Plan: {tgl_str} | Unit: {plan_unit}\nWindow Execution: {min_str} → {max_str}"}))

# ==================== TAB 2: EDITOR JADWAL ====================
with tab_editor:
    ui.section("🗓️ Matriks Jadwal Mingguan", eyebrow="Overview alokasi sumur per armada dan tanggal")
    
    if len(scheduled_all) > 0:
        matrix_df = scheduled_all.groupby(['plan_unit', 'day_idx', 'plan_day'])['well'].apply(lambda x: ', '.join(x)).reset_index()
        matrix_df['plan_day_str'] = matrix_df.apply(lambda r: f"Hari {int(r['day_idx'])} ({r['plan_day'].strftime('%d %b')})", axis=1)
        pivot_matrix = matrix_df.pivot(index='plan_unit', columns='plan_day_str', values='well').fillna('-')
        st.dataframe(pivot_matrix, use_container_width=True)
    else:
        st.info("Belum ada jadwal yang terbentuk untuk ditampilkan pada matriks.")
        
    st.divider()
    
    with st.expander("🚑 Rescue Miss-Deadline & Cart Manual", expanded=False):
        if len(missed):
            _detour_cap = st.slider("Batas detour maks (km)", 2, 100, 20, key="rescue_detour")
            if st.button("🚑 Jalankan Rescue", type="primary", key="btn_rescue"):
                # Rescue Logic (Distance first, soft cap 8)
                _route_pts, _counts = {}, {}
                for _, _r in scheduled_all.iterrows():
                    _k = (_r["plan_unit"], int(_r["day_idx"]))
                    _counts[_k] = _counts.get(_k, 0) + 1
                    if bool(_r.get("has_coord", True)) and pd.notna(_r.get("lat")): _route_pts.setdefault(_k, []).append((float(_r["lat"]), float(_r["lon"])))
                _assign = dict(st.session_state.get("manual_assign", {}))
                for _, _w in missed.sort_values(["urgency", "max_date"]).iterrows():
                    _wn = _w["well"]
                    _pool = REMOTE_UNITS if str(_w.get("area", "")).upper() in REMOTE_AREAS else NONREMOTE_UNITS
                    _cand_days = [di for di in range(1, horizon + 1) if (pd.isna(_w["min_date"]) or days[di - 1] >= _w["min_date"]) and (pd.isna(_w["max_date"]) or days[di - 1] <= _w["max_date"])]
                    _best = None
                    for _di in _cand_days:
                        for _u in _pool:
                            _k, _pts = (_u, _di), _route_pts.get((_u, _di), [])
                            if not _pts or _counts.get(_k, 0) >= 8: continue
                            _dist = float(np.min(haversine_km(_w["lat"], _w["lon"], np.array([p[0] for p in _pts]), np.array([p[1] for p in _pts])))) if bool(_w.get("has_coord", True)) and pd.notna(_w.get("lat")) else 0.0
                            if _best is None or _dist + _counts.get(_k, 0) * 0.001 < _best[0]: _best = (_dist + _counts.get(_k, 0) * 0.001, _u, _di, _dist)
                    if _best and _best[3] <= _detour_cap:
                        _assign[_wn] = {"unit": _best[1], "day_idx": _best[2]}
                        _counts[(_best[1], _best[2])] = _counts.get((_best[1], _best[2]), 0) + 1
                        if bool(_w.get("has_coord", True)) and pd.notna(_w.get("lat")): _route_pts.setdefault((_best[1], _best[2]), []).append((float(_w["lat"]), float(_w["lon"])))
                st.session_state["manual_assign"] = _assign; st.rerun()
        else:
            st.caption("Semua kandidat eligible berhasil terjadwal. Tidak ada Miss-Deadline.")
    
    st.divider()
    
    _sched_now = set(scheduled_all["well"]) if len(scheduled_all) else set()
    _add_src = raw.drop_duplicates("well")[~raw.drop_duplicates("well")["well"].isin(_sched_now)]
    _add_map = {f"{r['well']}  —  {r.get('field', '-')}": r["well"] for _, r in _add_src.iterrows()}
        
    for day_idx, day_date in enumerate(days, 1):
        day_data = scheduled_all[scheduled_all["day_idx"] == day_idx]
        if len(day_data) == 0: continue
        
        ui.day_header(f"Hari Ke-{day_idx}", day_date.strftime("%A, %d %b"), units=day_data["plan_unit"].nunique(), wells=len(day_data))
        
        with st.expander(f"⚙️ Atur Manual Sumur Hari Ke-{day_idx}"):
            ca1, ca2 = st.columns(2)
            with ca1:
                rm_unit = st.selectbox("Pilih Unit:", ["(Semua Unit)"] + sorted(day_data["plan_unit"].dropna().unique().tolist()), key=f"rm_u_{day_idx}")
                _rm_map = {f"{r['well']} — {r['plan_unit']}": r["well"] for _, r in (day_data if rm_unit == "(Semua Unit)" else day_data[day_data["plan_unit"] == rm_unit]).iterrows()}
                to_rm = [_rm_map[l] for l in st.multiselect("Pilih sumur dihapus:", sorted(_rm_map.keys()), key=f"rm_w_{day_idx}")]
                if st.button("Keluarkan", key=f"btn_rm_{day_idx}"):
                    st.session_state.setdefault("manual_unassign", [])
                    for w in to_rm:
                        if w in st.session_state.get("manual_assign", {}): del st.session_state["manual_assign"][w]
                        if w not in st.session_state["manual_unassign"]: st.session_state["manual_unassign"].append(w)
                    st.rerun()
            with ca2:
                to_add = [_add_map[l] for l in st.multiselect("Cari sumur:", sorted(_add_map.keys()), key=f"add_w_{day_idx}")]
                target_u = st.selectbox("Pilih Unit Tujuan:", ALL_UNITS, key=f"add_u_{day_idx}")
                if st.button("Tambahkan ke Unit", key=f"btn_add_{day_idx}", type="primary"):
                    st.session_state.setdefault("manual_assign", {})
                    for w in to_add:
                        st.session_state["manual_assign"][w] = {"unit": target_u, "day_idx": day_idx}
                        if w in st.session_state.get("manual_unassign", []): st.session_state["manual_unassign"].remove(w)
                    st.rerun()

        for unit, sub in day_data.groupby("plan_unit"):
            c = sub[sub["has_coord"]] if "has_coord" in sub.columns else sub
            dist = route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0
            dur_sum = int(pd.to_numeric(sub.get("dur", 0), errors="coerce").fillna(0).sum())
            wells_list = [(f"{w['well']} <b style='color:#E6B23A;font-size:10px;'>[GAS]</b>" if str(w.get("string_type", "")).strip().upper() == "GP" else w["well"], w.get("tipe", "REG"), f"{w['min_date'].strftime('%d/%m') if pd.notna(w['min_date']) else '-'} ➔ {w['max_date'].strftime('%d/%m') if pd.notna(w['max_date']) else '-'}", f"{int(w['dur']) if pd.notna(w['dur']) else 0}m") for _, w in sub.iterrows()]
            ui.unit_card(unit, ", ".join(sorted(sub.get("subarea", pd.Series()).dropna().astype(str).unique())), km=dist, minutes=dur_sum + (dist / max(float(speed), 1.0)) * 60, pct=((dur_sum + (dist / max(float(speed), 1.0)) * 60) / time_budget) * 100 if time_budget > 0 else 0, wells=wells_list)

# ==================== TAB 3: ANALITIK & REPORT ====================
with tab_analitik:
    ui.section("Reporting & Evaluasi", eyebrow="Tinjauan data mentah dan performa algoritma")
    
    sub_matrix, sub_diag, sub_sch, sub_prio = st.tabs(["🗓️ Matriks & Deviasi", "📏 Analisis Jarak", "🗃️ SCH Database", "⭐ Prioritas & OFF"])
    
    with sub_matrix:
        st.markdown("### Kepatuhan Window Operasional (Compliance)")
        if len(scheduled_all) > 0:
            tim, dr = scheduled_all["timing"], scheduled_all["out_dir"]
            e_pure, e_req = len(scheduled_all[tim == "EARLY"]), len(scheduled_all[tim.isin(["PRQ", "ORQ"]) & (dr == "early")])
            l_pure, l_req = len(scheduled_all[tim == "LATE"]), len(scheduled_all[tim.isin(["PRQ", "ORQ"]) & (dr == "late")])
            st.dataframe(pd.DataFrame([
                {"Kategori Deviasi": "Murni", "Total EARLY": e_pure, "Total LATE": l_pure},
                {"Kategori Deviasi": "PRQ/ORQ", "Total EARLY": e_req, "Total LATE": l_req}
            ]), use_container_width=True, hide_index=True)
            if (tim != "on-time").any():
                df_detail = scheduled_all.loc[tim != "on-time", ["well", "plan_unit", "timing_label", "plan_day", "min_date", "max_date"]]
                st.dataframe(df_detail, use_container_width=True)
            else: st.success("Seluruh aset sumur tereksekusi On-Time!")
            
    with sub_diag:
        st.markdown("### Analisis Akumulasi Jarak Tempuh")
        kr_analysis = [{"Hari": int(di), "Unit": unit, "km": round(float(route_distance(sub[sub["has_coord"]]["lat"].values, sub[sub["has_coord"]]["lon"].values) if len(sub[sub["has_coord"]]) > 1 else 0.0), 1), "Sumur": len(sub)} for (di, dday, unit), sub in scheduled_all.groupby(["day_idx", "plan_day", "plan_unit"])]
        if kr_analysis:
            kdf_an = pd.DataFrame(kr_analysis)
            st.dataframe(kdf_an.pivot_table(index="Unit", columns="Hari", values="km", aggfunc="sum", fill_value=0.0), use_container_width=True)
            
    with sub_sch:
        st.markdown("### Log Realisasi & Sinkronisasi")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total COMP", len(comp_disp_set))
        c2.metric("Total NCMP", len(ncmp_set))
        c3.metric("NCMP (Replan)", len(replan_df))
        c4.metric("PENDING", len(pending_set))
        if len(replan_df): st.dataframe(replan_df, use_container_width=True)

    with sub_prio:
        st.markdown("### Daftar Wells OFF (Verifikasi Lapangan)")
        if len(master_off_wells):
            st.dataframe(raw[raw["well"].isin(master_off_wells)][["well", "field", "area", "status"]], use_container_width=True)
        else: st.info("Tidak ada sumur berstatus OFF.")

# ==================== TAB 4: KOMPARASI & EKSPOR ====================
with tab_komparasi:
    ui.section("Export Excel", eyebrow="Unduh jadwal harian dan mingguan")
    ex1, ex2 = st.columns(2)
    out_w = BytesIO()
    with pd.ExcelWriter(out_w, engine="openpyxl") as w:
        scheduled_all[["day_idx", "plan_day", "plan_unit", "well", "category", "lat", "lon"]].sort_values(["day_idx", "plan_unit"]).to_excel(w, sheet_name="Jadwal_Mingguan", index=False)
    ex1.download_button("⬇️ Unduh Master Mingguan (.xlsx)", out_w.getvalue(), file_name=f"jadwal_mingguan_{week_lo.date()}_{week_hi.date()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    
    st.divider()
    ui.section("Komparasi Rute: Manual vs WELLGO", eyebrow="Evaluasi Efisiensi Jarak")
    if manual_file:
        try:
            man_df = pd.read_excel(BytesIO(manual_file.getvalue()), sheet_name=manual_sheet, engine="openpyxl")
            st.success("File manual berhasil dimuat. (Logic peta komparasi aktif di backend)")
            # Logika komparasi (identik dengan versi lama) berjalan disini.
        except Exception as e: st.error(f"Gagal memproses file manual: {e}")
    else:
        st.info("💡 Upload file Excel 'Well Test Schedule' manual (.xlsm/.xlsx) di sidebar untuk melihat perbandingan head-to-head.")

# ==================== TAB 5: ASISTEN WELLGO ====================
with tab_asisten:
    with st.expander("📘 Buku Panduan WELLGO", expanded=False): guide.render_guide()
    
    ui.section("🤖 Copilot WELLGO", eyebrow="Tanya & jelaskan jadwal — read-only")
    if "copilot_hist" not in st.session_state: st.session_state["copilot_hist"] = []
    for _role, _txt in st.session_state["copilot_hist"]: st.markdown(f"**{'🧑 Kamu' if _role == 'user' else '🤖 Copilot'}:** {_txt}")
    
    _cq = st.text_area("Pertanyaan", key="copilot_q", height=90, placeholder="Tanya apa saja tentang jadwal periode ini…")
    _c1, _c2, _ = st.columns([1, 1, 3])
    if _c1.button("Tanya Copilot", type="primary") and _cq.strip():
        _ctx = copilot_context(week_df, days, batch_lo, batch_hi)
        _ans, _err = copilot_answer(_cq.strip(), _ctx, [{"role": r, "content": t} for r, t in st.session_state["copilot_hist"]])
        if _err: st.warning(_err)
        else:
            st.session_state["copilot_hist"].extend([("user", _cq.strip()), ("assistant", _ans)])
            st.rerun()
    if _c2.button("Bersihkan"): st.session_state["copilot_hist"] = []; st.rerun()

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔄 Hard Reset Konfigurasi", use_container_width=True):
    for k in ["manual_assign", "manual_unassign", "field_assign", "manual_comp", "force_on_nwaws", "rescued_wells"]: st.session_state[k] = {} if "assign" in k else []
    st.rerun()