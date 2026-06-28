"""
WELLGO (Well Grouping Optimizer) — 9 Unit MWT Daily Planner
----------------------------------------------------------
Optimasi rute logistik well testing Sumatra Light North (SL North).
Terintegrasi dengan design system `wellgo_ui`.

Run: py -m streamlit run app.py
"""

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

def status_in_period(lo, hi):
    con = sqlite3.connect(DB_PATH)
    try:
        q = ("SELECT well_name AS well, status, reason, plan_date FROM execution_log "
             "WHERE status IN ('executed','ncmp') AND plan_date BETWEEN ? AND ?")
        df = pd.read_sql(q, con, params=(str(lo), str(hi)))
    except Exception:
        df = pd.DataFrame(columns=["well", "status", "reason", "plan_date"])
    con.close()
    if not len(df):
        return set(), df[["well", "reason", "plan_date"]] if "well" in df else pd.DataFrame(
            columns=["well", "reason", "plan_date"])
    df["plan_date"] = df["plan_date"].astype(str)
    latest = df[df["plan_date"] == df.groupby("well")["plan_date"].transform("max")]
    flag = latest.groupby("well")["status"].apply(lambda s: (s == "executed").any())
    executed = set(flag[flag].index)
    ncmp_w = set(flag[~flag].index)
    ncmp = (latest[latest["well"].isin(ncmp_w)]
            .sort_values("plan_date").groupby("well", as_index=False).last()[["well", "reason", "plan_date"]])
    return executed, ncmp

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
    n_comp = n_ncmp = 0
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
            except Exception:
                continue
            if {"WELL", "STATUS", "SCHEDULE_DATE_TEST"} <= up:
                sht = s
                break
        if sht is None:
            sht = next((s for s in xls.sheet_names if s.strip().upper().replace(" ", "").replace("_", "")
                        in ("SCHDATABASE", "COMPNCMP", "SCHSTATUS")), xls.sheet_names[0])
        df = pd.read_excel(xls, sheet_name=sht)
        cols = {str(c).strip().upper(): c for c in df.columns}
        cw = cols.get("WELL")
        cs = cols.get("STATUS")
        cd = cols.get("SCHEDULE_DATE_TEST")
        cu = cols.get("UNIT")
        cr = cols.get("COMMENT IF NOT COMPLETE")
        if not (cw and cs and cd): continue
        w = pd.DataFrame({
            "well": df[cw].astype(str).str.strip(),
            "raw_stat": df[cs].astype(str).str.strip().str.upper(),
            "date": pd.to_datetime(df[cd], errors="coerce"),
            "unit": df[cu].map(norm_unit) if cu else "",
            "reason": (df[cr].astype(str).str.strip().str.upper().replace({"NAN": ""}) if cr else ""),
        })
        for k, v in w["raw_stat"].value_counts().items():
            status_seen[k] = status_seen.get(k, 0) + int(v)
        w["stat"] = w["raw_stat"].map(classify_status)
        skip_well += int(w["well"].isin(["", "nan"]).sum())
        skip_status += int((w["stat"] == "").sum())
        skip_date += int(w["date"].isna().sum())
        w = w[(~w["well"].isin(["", "nan"])) & (w["stat"] != "") & (w["date"].notna())].copy()
        w["plan_date"] = w["date"].dt.date.astype(str)
        w["log_status"] = np.where(w["stat"] == "COMP", "executed", "ncmp")
        rows = list(zip(w["plan_date"], w["well"], w["unit"], w["log_status"], w["reason"],
                        [now] * len(w)))
        con.executemany("""INSERT INTO execution_log(plan_date,well_name,unit,status,reason,updated_at)
            VALUES(?,?,?,?,?,?) ON CONFLICT(plan_date,well_name) DO UPDATE SET
            unit=excluded.unit, status=excluded.status, reason=excluded.reason,
            updated_at=excluded.updated_at""", rows)
        n_comp += int((w["stat"] == "COMP").sum())
        nc = w[w["stat"] == "NCMP"]
        n_ncmp += len(nc)
        for rsn, cnt in nc["reason"].replace("", "(kosong)").value_counts().items():
            reasons[rsn] = reasons.get(rsn, 0) + int(cnt)
    con.commit()
    con.close()
    return {"comp": n_comp, "ncmp": n_ncmp, "reasons": reasons, "status_seen": status_seen,
            "skip_date": skip_date, "skip_well": skip_well, "skip_status": skip_status}

def save_coords(pairs):
    con = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat(timespec="seconds")
    for well, lat, lon in pairs:
        if pd.notna(lat) and pd.notna(lon):
            con.execute("""INSERT INTO coord_cache(well_name,lat,lon,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(well_name) DO UPDATE SET lat=excluded.lat, lon=excluded.lon,
                updated_at=excluded.updated_at""", (well, float(lat), float(lon), now))
    con.commit()
    con.close()

def load_coord_cache():
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT well_name,lat,lon FROM coord_cache", con)
    except Exception:
        df = pd.DataFrame(columns=["well_name", "lat", "lon"])
    con.close()
    return df

# ------------------------------------------------------------------ data
def to_dt(col):
    num = pd.to_numeric(col, errors="coerce")
    valid = num.dropna()
    if len(valid) and valid.between(20000, 60000).mean() > 0.5:
        return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(col, errors="coerce")

@st.cache_data(show_spinner=False)
def load_spatial_data(file_bytes, sheet):
    try:
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
        df.columns = [str(c).strip().upper() for c in df.columns]
        req_cols = {"WELL", "FIELD", "LAT", "LON"}
        if not req_cols.issubset(set(df.columns)): return pd.DataFrame()
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        df = df.dropna(subset=["WELL", "LAT", "LON"])
        df = df.drop_duplicates(subset=["WELL"], keep="first")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_candidates(file_bytes, sheet):
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    ren = {
        "well_name": "well", "Surface Lat": "lat", "Surface Lon": "lon",
        "Duration test (minutes)": "dur", "min_execution date": "min_date",
        "max_execution_date": "max_date", "op_sub_area_code": "subarea",
        "op_area_code": "area", "test_category": "category",
        "well_tier": "tier", "field": "field", "string_type": "string_type",
        "Remark": "remark", "REMARK for IEMS Req or Spare candidate": "remark_iems"}

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

    np_col = next((c for c in df.columns if "NEXT" in str(c).upper()
                   and ("PROPOS" in str(c).upper() or "WT" in str(c).upper())), None)
    df["next_wt"] = to_dt(df[np_col]) if np_col else pd.NaT
    df["unit"] = df["unit"].map(norm_unit_name)

    st_ = df["string_type"].astype(str).str.upper().str.strip()
    area_ = df["area"].astype(str).str.upper().str.strip()
    fld_ = df["field"].astype(str).str.upper().str.strip()
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
    cat_force = cat_u.isin(NWAWS)
    df["is_nwaws"] = cat_force
    df["tipe"] = np.where(cat_u.str.contains("NEW WELL"), "NW",
                          np.where(cat_u.str.contains("AWS"), "AWS", "REG"))
    rmk = (df["remark"].astype(str).fillna("") + " " + df["remark_iems"].astype(str).fillna("")).str.upper()
    is_req = rmk.str.contains("REQ", na=False) | rmk.str.contains("DEEPENING", na=False)
    df["force_week"] = cat_force | is_req
    ops_req = is_req & rmk.str.contains("OPS", na=False)
    df["req_tag"] = np.where(ops_req, "ORQ", np.where(is_req, "PRQ", ""))

    status_col = next((c for c in df.columns if str(c).strip().upper() in ("WELL STATUS", "LAST_STATUS")), None)
    df["status"] = (df[status_col].astype(str).str.upper().str.strip() if status_col else "ON")
    sch_col = next((c for c in df.columns if str(c).strip().upper() in ("SCH STATUS", "SCH_STATUS")), None)
    df["sch_status"] = (df[sch_col].astype(str).str.upper().str.strip() if sch_col else "")
    df["sch_status"] = df["sch_status"].replace({"NAN": "", "NONE": ""})
    return df

def good_coord(lat, lon):
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    return pd.notna(lat) & pd.notna(lon) & lat.between(0.1, 5) & lon.between(95, 110)

def resolve_coords(df, spatial_db, cache, field_assign=None):
    df = df.copy()
    field_assign = field_assign or {}
    df["coord_source"] = "none"

    if not spatial_db.empty:
        s_map = spatial_db.set_index("WELL")
        has_master = df["well"].isin(s_map.index)
        df.loc[has_master, "lat"] = df.loc[has_master, "well"].map(s_map["LAT"])
        df.loc[has_master, "lon"] = df.loc[has_master, "well"].map(s_map["LON"])

        empty_fld = df["field"].isna() | (df["field"] == "")
        upd_fld = empty_fld & has_master
        if "FIELD" in s_map.columns:
            df.loc[upd_fld, "field"] = df.loc[upd_fld, "well"].map(s_map["FIELD"])

        df.loc[has_master & good_coord(df["lat"], df["lon"]), "coord_source"] = "master_spasial"

    cand_good = good_coord(df["lat"], df["lon"]) & (df["coord_source"] == "none")
    df.loc[cand_good, "coord_source"] = "database"

    if not cache.empty:
        cmap = cache.set_index("well_name")
        miss = df["coord_source"] == "none"
        has_cache = miss & df["well"].isin(cmap.index)
        df.loc[has_cache, "lat"] = df.loc[has_cache, "well"].map(cmap["lat"])
        df.loc[has_cache, "lon"] = df.loc[has_cache, "well"].map(cmap["lon"])
        df.loc[has_cache, "coord_source"] = "cache"

    if not spatial_db.empty and "FIELD" in spatial_db.columns:
        cent_f = spatial_db.groupby("FIELD")[["LAT", "LON"]].mean()
    else:
        base = df[df["coord_source"].isin(["master_spasial", "database", "cache"])]
        cent_f = base.groupby("field")[["lat", "lon"]].mean() if len(base) else pd.DataFrame(columns=["lat", "lon"])
        cent_f.columns = ["LAT", "LON"]

    miss = df["coord_source"] == "none"
    has_cent = miss & df["field"].isin(cent_f.index)
    df.loc[has_cent, "lat"] = df.loc[has_cent, "field"].map(cent_f["LAT"])
    df.loc[has_cent, "lon"] = df.loc[has_cent, "field"].map(cent_f["LON"])
    df.loc[has_cent, "coord_source"] = "imputed_field"

    for well, fld in field_assign.items():
        m = (df["well"] == well) & (df["coord_source"] == "none")
        if m.any() and fld in cent_f.index:
            df.loc[m, "field"] = fld
            df.loc[m, "lat"] = cent_f.loc[fld, "LAT"]
            df.loc[m, "lon"] = cent_f.loc[fld, "LON"]
            df.loc[m, "coord_source"] = "manual_field"

    df["has_coord"] = df["coord_source"] != "none"
    return df

# ------------------------------------------------------------------ geometry
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def _haversine_matrix(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    p = np.pi / 180.0
    la = lat[:, None] * p
    lo = lon[:, None] * p
    a = 0.5 - np.cos(la.T - la) / 2.0 + np.cos(la) * np.cos(la.T) * (1.0 - np.cos(lo.T - lo)) / 2.0
    a = np.clip(a, 0.0, 1.0)
    d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    return np.nan_to_num(d, nan=1e9, posinf=1e9)

def _solve_route(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = len(lat)
    if n <= 1: return list(range(n)), 0.0
    D = _haversine_matrix(lat, lon)
    if n == 2: return [0, 1], float(D[0, 1])
    dist = D.tolist()

    best_order, best_total = None, float("inf")
    rng = range(n)
    for start in rng:
        used = [False] * n
        used[start] = True
        order = [start]
        for _ in range(n - 1):
            drow = dist[order[-1]]
            bd = float("inf")
            bn = -1
            for j in rng:
                if not used[j]:
                    v = drow[j]
                    if v < bd:
                        bd = v
                        bn = j
            order.append(bn)
            used[bn] = True

        improved = True
        while improved and n > 3:
            improved = False
            for i in range(1, n - 2):
                oi1 = order[i - 1]
                oi = order[i]
                for j in range(i + 2, n):
                    n3 = order[j - 1]
                    n4 = order[j]
                    if dist[oi1][n3] + dist[oi][n4] + 1e-9 < dist[oi1][oi] + dist[n3][n4]:
                        order[i:j] = order[i:j][::-1]
                        improved = True
                        oi = order[i]

        total = float(sum(dist[order[k]][order[k + 1]] for k in range(n - 1)))
        if total < best_total:
            best_total, best_order = total, order
    return best_order, best_total

@st.cache_resource(show_spinner=False)
def _route_cache_store():
    return {}

def route_distance(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size <= 1: return 0.0
    m = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[m], lon[m]
    if lat.size <= 1: return 0.0
    key = tuple(sorted(zip(np.round(lat, 5).tolist(), np.round(lon, 5).tolist())))
    store = _route_cache_store()
    val = store.get(key)
    if val is None:
        val = _solve_route(lat, lon)[1]
        if len(store) > 50000: store.clear()
        store[key] = val
    return val

def optimize_route(lat, lon):
    return _solve_route(lat, lon)

def convex_hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2: return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
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
    lat = pd.to_numeric(sub["lat"], errors="coerce").values
    lon = pd.to_numeric(sub["lon"], errors="coerce").values
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
        dlat = (r / 111.0) * math.sin(a)
        dlon = (r / (111.0 * math.cos(math.radians(clat)))) * math.cos(a)
        out.append([clon + dlon, clat + dlat])
    return out

# ------------------------------------------------------------------ engine
def plan(elig, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, current_day=None, elastic_limit=5.0):
    df = elig.reset_index(drop=True).copy()
    df["scheduled"] = False
    df["plan_unit"] = None
    if "forced_unit" not in df.columns: df["forced_unit"] = None
    if "urgency" not in df.columns: df["urgency"] = 0.0

    lats = pd.to_numeric(df["lat"], errors="coerce").values
    lons = pd.to_numeric(df["lon"], errors="coerce").values
    dist_mat = _haversine_matrix(lats, lons)

    field_arr = df["field"].values
    area_arr = df["area"].values
    urg_arr = pd.to_numeric(df["urgency"], errors="coerce").fillna(0).values
    dur_arr = pd.to_numeric(df["dur"], errors="coerce").fillna(0).values
    speed = max(float(speed), 1.0)

    if mode == "dedicated":
        for unit in df["unit"].dropna().unique():
            idxs = list(df.index[df["unit"] == unit])
            if not idxs: continue
            ordered = sorted(idxs, key=lambda i: (urg_arr[i], dur_arr[i]))
            sel = ordered[:max_wells]
            if use_dur:
                while len(sel) > 1:
                    dist = route_distance(lats[sel], lons[sel])
                    if dur_arr[sel].sum() + (dist / speed) * 60 <= time_budget: break
                    sel = sorted(sel, key=lambda i: urg_arr[i])[:-1]
            df.loc[sel, "scheduled"] = True
            df.loc[sel, "plan_unit"] = unit
        return df

    avail_remote = list(REMOTE_UNITS)[:n_remote]
    avail_nonremote = list(NONREMOTE_UNITS)[:n_nonremote]
    unit_clusters = {u: [] for u in avail_remote + avail_nonremote}
    unassigned = set(df.index)

    fu_mask = df["forced_unit"].notna() & df["forced_unit"].isin(unit_clusters.keys())
    for u, grp in df[fu_mask].groupby("forced_unit"):
        grp_sorted = grp.sort_values(["urgency", "dur"])
        for idx in grp_sorted.index:
            if len(unit_clusters[u]) < max_wells and idx in unassigned:
                if unit_clusters[u]:
                    d_min = dist_mat[unit_clusters[u], idx].min()
                    d_max = dist_mat[unit_clusters[u], idx].max()
                    urg_idx = urg_arr[idx]
                    
                    if d_max > elastic_limit:
                        continue
                        
                    if d_min > 5.0 and not (d_min <= elastic_limit and urg_idx <= 2): continue
                if use_dur and unit_clusters[u]:
                    cand = unit_clusters[u] + [idx]
                    dist = route_distance(lats[cand], lons[cand])
                    if dur_arr[cand].sum() + (dist / speed) * 60 > time_budget: continue
                unit_clusters[u].append(idx)
                unassigned.remove(idx)

    used_units = {u for u, c in unit_clusters.items() if len(c) > 0}
    avail_remote = [u for u in avail_remote if u not in used_units]
    avail_nonremote = [u for u in avail_nonremote if u not in used_units]

    while unassigned and (avail_remote or avail_nonremote):
        field_scores = {}
        un_list = list(unassigned)
        un_fields = field_arr[un_list]

        for fld in pd.unique(un_fields):
            f_wells = [w for w in un_list if field_arr[w] == fld]
            if not f_wells: continue
            urgs = urg_arr[f_wells]
            score = 0
            for u in urgs:
                if u < -1000: score += 100000
                elif u <= 0: score += 10000
                elif u == 1: score += 5000
                elif u == 2: score += 1000
                elif u <= 4: score += 100
                else: score += 10
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
            f_wells_sorted = sorted(f_wells, key=lambda x: (urg_arr[x], dur_arr[x]))
            seed = f_wells_sorted[0]

            unit_clusters[u].append(seed)
            unassigned.remove(seed)
            used_units.add(u)

            while len(unit_clusters[u]) < max_wells and unassigned:
                cand_pool = [i for i in unassigned if (area_arr[i] in REMOTE_AREAS) == (zone == "remote")]
                if not cand_pool: break

                c_dists = dist_mat[np.ix_(unit_clusters[u], cand_pool)]
                min_dists = c_dists.min(axis=0)
                max_dists = c_dists.max(axis=0)

                valid_cands = []
                for i_cand, cand_idx in enumerate(cand_pool):
                    d_min = min_dists[i_cand]
                    d_max = max_dists[i_cand]
                    urg = urg_arr[cand_idx]
                    is_same_fld = (field_arr[cand_idx] == target_fld)
                    
                    if d_max > elastic_limit:
                        continue
                        
                    if is_same_fld or d_min <= 5.0 or (d_min <= elastic_limit and urg <= 2):
                        c_score = d_min
                        if is_same_fld: c_score -= 50
                        if urg <= 2: c_score -= 20
                        valid_cands.append((cand_idx, d_min, c_score))

                if not valid_cands: break
                valid_cands.sort(key=lambda x: x[2])

                best_idx = None
                for cand_idx, d, _ in valid_cands:
                    cand_cluster = unit_clusters[u] + [cand_idx]
                    if use_dur:
                        dist = route_distance(lats[cand_cluster], lons[cand_cluster])
                        if dur_arr[cand_cluster].sum() + (dist / speed) * 60 <= time_budget:
                            best_idx = cand_idx
                            break
                    else:
                        best_idx = cand_idx
                        break

                if best_idx is not None:
                    unit_clusters[u].append(best_idx)
                    unassigned.remove(best_idx)
                else:
                    break

            assigned_this_round = True
            break
        if not assigned_this_round: break

    for u, c in unit_clusters.items():
        if c:
            df.loc[c, "scheduled"] = True
            df.loc[c, "plan_unit"] = u

    return df

def plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed,
              use_urg, use_dur, early_days=0, elastic_limit=5.0):
    elig = elig.reset_index(drop=True).copy()
    elig["scheduled"] = False
    elig["plan_unit"] = None
    elig["plan_day"] = pd.NaT
    elig["day_idx"] = 0
    early_td = pd.Timedelta(days=early_days)
    rem = pd.Series(True, index=elig.index)

    is_nwaws = elig["tipe"].isin(["NW", "AWS"])
    fw_c = elig["force_week"].fillna(False) if "force_week" in elig.columns else pd.Series(False, index=elig.index)
    cc_c = elig["carry_ncmp"].fillna(False) if "carry_ncmp" in elig.columns else pd.Series(False, index=elig.index)
    
    bypass_reg = (fw_c & ~is_nwaws) | (cc_c & ~is_nwaws)

    np_in = elig["np_in_range"].fillna(False) if "np_in_range" in elig.columns else pd.Series(False, index=elig.index)
    next_wt = elig["next_wt"] if "next_wt" in elig.columns else pd.Series(pd.NaT, index=elig.index)
    
    strict_no_late = elig["np_in_range"] & elig["max_in_range"] & ~is_nwaws

    for i, day in enumerate(days, start=1):
        win_reg = (elig["min_date"] - early_td <= day)
        win_nw = (elig["min_date"] <= day) & (elig["max_date"] >= day)
        np_ok = np_in & (next_wt <= day) & ~is_nwaws

        is_late = day > elig["max_date"]
        forbid_late = strict_no_late & is_late

        cond_reg = (~is_nwaws) & (win_reg | np_ok | bypass_reg) & ~forbid_late
        cond_nw = is_nwaws & win_nw
        
        pidx = elig.index[rem & (cond_reg | cond_nw)]
        if len(pidx) == 0: continue

        pool = elig.loc[pidx].copy()
        pool["urgency"] = (pool["max_date"] - day).dt.days.fillna(0)
        
        mid = bypass_reg.loc[pidx]
        nw = is_nwaws.loc[pidx]

        pool.loc[mid, "urgency"] = pool.loc[mid, "urgency"].clip(upper=0)
        pool.loc[nw, "urgency"] = pool.loc[nw, "urgency"].clip(upper=0) - 10000

        pd_ = plan(pool, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, current_day=day, elastic_limit=elastic_limit)

        sd = pd_[pd_["scheduled"]]
        if len(sd) == 0: continue

        sidx = elig.index[elig["well"].isin(sd["well"])]
        elig.loc[sidx, "scheduled"] = True
        elig.loc[sidx, "plan_day"] = day
        elig.loc[sidx, "day_idx"] = i
        elig.loc[sidx, "plan_unit"] = elig.loc[sidx, "well"].map(dict(zip(sd["well"], sd["plan_unit"])))
        rem.loc[sidx] = False

    elig["urgency"] = (elig["max_date"] - days[0]).dt.days
    return elig

def unit_summary(df, speed):
    cols = ["Unit", "Sumur", "Test (min)", "Rute (km)", "Est (min)",
            "Sub-area", "Deadline tercepat", "⏱️ Early/Late", "Wells"]
    if df is None or not len(df) or "scheduled" not in df.columns:
        return pd.DataFrame(columns=cols)
    sched = df[df["scheduled"].fillna(False)]
    if not len(sched) or "plan_unit" not in sched.columns:
        return pd.DataFrame(columns=cols)

    speed = max(float(speed), 1.0)
    rows = []
    for unit, sub in sched.groupby("plan_unit"):
        if "has_coord" in sub.columns:
            c = sub[sub["has_coord"].fillna(False)]
        else:
            c = sub
        dist = route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0

        notes = []
        if "timing_label" in sub.columns:
            notes = [f"{w} ({lab})" for w, lab in zip(sub["well"], sub["timing_label"]) if lab]

        dur_sum = int(pd.to_numeric(sub["dur"], errors="coerce").fillna(0).sum()) if "dur" in sub.columns else 0
        dmin = sub["max_date"].min() if "max_date" in sub.columns else pd.NaT
        deadline = dmin.strftime("%Y-%m-%d") if pd.notna(dmin) else "-"
        subarea = ", ".join(sorted(sub["subarea"].dropna().astype(str).unique())) if "subarea" in sub.columns else ""

        rows.append({
            "Unit": unit, "Sumur": len(sub), "Test (min)": dur_sum,
            "Rute (km)": round(dist, 1), "Est (min)": int(dur_sum + (dist / speed) * 60),
            "Sub-area": subarea,
            "Deadline tercepat": deadline,
            "⏱️ Early/Late": ", ".join(notes) or "-",
            "Wells": ", ".join(sub["well"].astype(str))})
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("Unit")

COLORS = [[228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0],
          [166, 86, 40], [247, 129, 191], [26, 188, 156], [241, 196, 15], [106, 61, 154],
          [178, 223, 138], [251, 154, 153]]

def cmap(label, labels):
    try: return COLORS[list(labels).index(label) % len(COLORS)]
    except ValueError: return [130, 130, 130]


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
    up = st.file_uploader("Upload Excel kandidat & spasial", type=["xlsx", "xlsm"],
                          help="File Excel berisi sheet Kandidat Sumur & Data_Spasial (Master Database Koordinat)")
    
    col_sh1, col_sh2 = st.columns(2)
    with col_sh1: sheet_kandidat = st.text_input("Sheet Kandidat", SHEET_DEFAULT)
    with col_sh2: sheet_spasial = st.text_input("Sheet Spasial", "Data_Spasial")
        
    mpas_only = st.checkbox("Hanya Unit Tes (MPAS), exclude TS", value=True)

if up is None:
    ui.hero_header(date_str=datetime.now().strftime("%d %b %Y"), horizon=7, units=9, compliance=0, mode="pooled")

    t_mulai, t_panduan = st.tabs(["🏠 Mulai", "📘 Panduan"])
    with t_mulai:
        st.info("💡 Silakan unggah berkas Excel data kandidat sumur & master database koordinat "
                "spasial pada sidebar untuk memulai kalkulasi rute.")
    with t_panduan:
        guide.render_guide()

    st.stop()


# ── Data Loading Awal untuk Filter Area ─────────────────────────────────────
raw = load_candidates(up.getvalue(), sheet_kandidat)
spatial_db = load_spatial_data(up.getvalue(), sheet_spasial)

if spatial_db.empty: st.error("⚠️ Struktur berkas Data Spasial tidak valid atau kosong. Pastikan sheet mengandung kolom: WELL, FIELD, LAT, LON.")

# Meneruskan Sidebar...
with st.sidebar:
    st.divider()
    ui.section("🗺️ Filter Area & Unit")
    all_areas = sorted(raw["area"].dropna().unique())
    default_excl = [a for a in all_areas if a == "LIBO"]
    excl_areas = st.multiselect("Exclude Area Terpilih", all_areas, default=default_excl)
    if mpas_only:
        ts_unavail = st.multiselect("Area Fasilitas TS Down (Dialihkan ke MWT)", all_areas)
        mwt_unavail = st.multiselect("Area Fleet MWT Down (Dialihkan ke TS)", all_areas)
    else:
        ts_unavail, mwt_unavail = [], []
    
    st.divider()
    ui.section("⏱️ Status Realisasi Harian")
    comp_files = st.file_uploader("Upload file COMP/NCMP harian", type=["xlsx", "xlsm"], accept_multiple_files=True)
    skip_woff = st.checkbox("Skip sumur NCMP yang berstatus OFF", value=True)
    
    st.divider()
    
    ui.section("📅 Horizon Perencanaan")
    _today = datetime.now().date()
    periode = st.date_input("Rentang Siklus (Periode)", value=(_today, _today + timedelta(days=6)),
                            help="Batas siklus keseluruhan. Menentukan data NCMP yang dibaca & kelayakan sumur.")
    
    if isinstance(periode, (list, tuple)) and len(periode) == 2:
        per_lo, per_hi = periode[0], periode[1]
    else:
        per_lo = periode if not isinstance(periode, (list, tuple)) else periode[0]
        per_hi = per_lo + timedelta(days=6)

    plan_start_date = st.date_input("Mulai Planning dari Tanggal",
                                    value=per_lo, min_value=per_lo, max_value=per_hi,
                                    help="Titik mulai optimasi rute. Mengikuti rentang siklus di atas.")
                                    
    st.divider()
    
    with st.form("opt_form"):
        ui.section("⚙️ Parameter Algoritma")
        mode_label = st.radio("Mode Distribusi Unit", ["Dedicated (Territory)", "Pooled (Bebas Zona)"], index=1)
        mode = "dedicated" if mode_label.startswith("Dedicated") else "pooled"
        crit_label = st.radio("Kriteria Utama Optimasi", list(CRIT.keys()), index=3)
        use_urg, use_dur = CRIT[crit_label]

        st.divider()
        max_wells = st.slider("Target Sumur / Unit / Hari", 3, 8, 6)
        ded = (mode == "dedicated")
        n_remote = st.slider("Unit Area Remote (Bangko/Balam)", 1, 5, 5, disabled=ded)
        n_nonremote = st.slider("Unit Area Non-Remote (Bekasap)", 1, 4, 4, disabled=ded)
        
        elastic_limit = st.slider("Batas Persebaran Rute (Elastic Limit km)", 5, 50, 5, 1, help="Mencegah efek 'chaining' di mana armada merangkai jarak dekat tapi ujung ke ujungnya terlalu jauh.")
        
        time_budget = st.slider("Time Budget / Hari (Menit)", 180, 540, 360, 30, disabled=not use_dur)
        speed = st.slider("Kecepatan Rata-rata Fleet (km/jam)", 10, 60, 25, 5)
        early_days = st.slider("Skenario Early Test (H-Min)", 0, 7, 0)
        show_block = st.checkbox("Tampilkan Block Area Field di Peta", value=True)
        
        submit_btn = st.form_submit_button("🔄 Re-run Optimizer", type="primary", use_container_width=True)

# Parameter Kalkulasi
per_lo_ts, per_hi_ts = pd.Timestamp(per_lo), pd.Timestamp(per_hi)
plan_start_ts = pd.Timestamp(plan_start_date)
horizon = (per_hi_ts - plan_start_ts).days + 1
if horizon < 1: horizon = 1
if horizon > 60: horizon = 60

# ── Data Processing Block ──────────────────────────────────────────────────
if comp_files:
    import hashlib
    sig = hashlib.md5(b"".join(sorted(f.getvalue() for f in comp_files))).hexdigest()
    if st.session_state.get("_compncmp_sig") != sig:
        with st.spinner("Sinkronisasi status COMP/NCMP harian..."):
            summ_imp = import_compncmp([f.getvalue() for f in comp_files])
        st.session_state["_compncmp_sig"] = sig
        st.session_state["_compncmp_summary"] = summ_imp
    summ_imp = st.session_state["_compncmp_summary"]

if excl_areas: raw = raw[~raw["area"].isin(excl_areas)].copy()

ts_redirected = raw[raw["is_ts"] & raw["area"].isin(ts_unavail)].copy()
mwt_redirected = raw[raw["is_mpas"] & raw["area"].isin(mwt_unavail)].copy()
ts_wells = raw[raw["is_ts"] & ~raw["area"].isin(ts_unavail)].copy()
if mpas_only:
    plannable = (((raw["is_mpas"] | raw["unit_unknown"]) & ~raw["area"].isin(mwt_unavail))
                 | (raw["is_ts"] & raw["area"].isin(ts_unavail)))
    raw = raw[plannable].copy()

field_assign = st.session_state.get("field_assign", {})
raw = resolve_coords(raw, spatial_db, load_coord_cache(), field_assign=field_assign)

if not spatial_db.empty and "FIELD" in spatial_db.columns:
    field_list = sorted(spatial_db["FIELD"].dropna().unique().tolist())
    field_wells_coord = spatial_db.rename(columns={"WELL": "well", "FIELD": "field", "LAT": "lat", "LON": "lon"})
else:
    _basecoord = raw[raw["coord_source"].isin(["master_spasial", "database", "cache"])]
    field_centroids = (_basecoord.groupby("field")[["lat", "lon"]].mean() if len(_basecoord) else pd.DataFrame(columns=["lat", "lon"]))
    field_list = sorted(field_centroids.index.tolist())
    field_wells_coord = _basecoord[["field", "well", "lat", "lon"]].copy()

days = [plan_start_ts + pd.Timedelta(days=i) for i in range(horizon)]
week_lo, week_hi = days[0], days[-1]

executed_log, ncmp_log = status_in_period(per_lo, per_hi)
comp_col = set(raw.loc[raw["sch_status"] == "COMP", "well"])
executed = executed_log | comp_col

ncmp_log = ncmp_log[~ncmp_log["well"].isin(executed)].copy()
ncmp_col = set(raw.loc[raw["sch_status"] == "NCMP", "well"]) - executed
ncmp_set = (set(ncmp_log["well"]) | ncmp_col) - executed

in_raw = set(raw["well"])
master_off_wells = set(raw.loc[raw["status"] == "OFF", "well"])

if skip_woff:
    woff_set = ncmp_set & master_off_wells
else:
    woff_set = set()

ncmp_replan = (ncmp_set & in_raw) - woff_set
ncmp_no_data = sorted(ncmp_set - in_raw)

ncmp_col_df = pd.DataFrame({"well": sorted(ncmp_col), "reason": "", "plan_date": "(kolom)"})
ncmp_df = pd.concat([ncmp_log, ncmp_col_df], ignore_index=True).drop_duplicates("well")
replan_df = ncmp_df[ncmp_df["well"].isin(ncmp_replan)].copy()

batch_lo, batch_hi = per_lo_ts, per_hi_ts
win_in_range = (raw["min_date"] <= batch_hi) & (raw["max_date"] >= batch_lo)
np_in_range = (raw["next_wt"] >= batch_lo) & (raw["next_wt"] <= batch_hi)
in_range = win_in_range | np_in_range
is_nwaws_c = raw["is_nwaws"].fillna(False)
req_force = raw["force_week"].fillna(False) & ~is_nwaws_c
is_ncmp = raw["well"].isin(ncmp_replan)
comp_wells = raw[raw["well"].isin(executed)].copy()
nwaws_dropped = raw[is_nwaws_c & ~in_range & (~raw["well"].isin(executed))].copy()
cand = raw[(in_range | is_ncmp | req_force) & (~raw["well"].isin(executed))].copy()

cand["np_in_range"] = np_in_range.loc[cand.index]
cand["max_in_range"] = ((raw["max_date"] >= batch_lo) & (raw["max_date"] <= batch_hi)).loc[cand.index]

off_wells = cand[cand["status"] == "OFF"].copy()
woff_wells = raw[raw["well"].isin(woff_set)].copy()
elig_all = cand[(cand["status"] != "OFF") & (~cand["well"].isin(woff_set))].copy()
elig_all["carry_ncmp"] = elig_all["well"].isin(ncmp_replan)

elig_all["urgency"] = (elig_all["max_date"] - week_lo).dt.days
elig_all["urgency"] = elig_all["urgency"].fillna(0)

nwaws = elig_all["is_nwaws"].fillna(False)
mid_prio = (elig_all["force_week"].fillna(False) & ~nwaws) | elig_all["carry_ncmp"]

elig_all.loc[mid_prio, "urgency"] = elig_all.loc[mid_prio, "urgency"].clip(upper=0)
elig_all.loc[nwaws, "urgency"] = elig_all.loc[nwaws, "urgency"].clip(upper=0) - 10000

elig = elig_all[elig_all["has_coord"]].copy()
nocoord = elig_all[~elig_all["has_coord"]].copy()

# ── Rollout Execution Framework ────────────────────────────────────────────
if len(elig):
    week_df = plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit)
else:
    week_df = elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
if len(nocoord):
    noc = nocoord.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
    week_df = pd.concat([week_df, noc], ignore_index=True)

week_df["zone"] = np.where(week_df["area"].isin(REMOTE_AREAS), "remote", "non-remote")
week_df["manual"] = False

man = st.session_state.get("manual_assign", {})
if man:
    for w, info in list(man.items()):
        m = week_df["well"] == w
        if not m.any(): continue
        di = int(info["day_idx"])
        if di < 1 or di > horizon: continue
        week_df.loc[m, "scheduled"] = True
        week_df.loc[m, "plan_unit"] = info["unit"]
        week_df.loc[m, "day_idx"] = di
        week_df.loc[m, "plan_day"] = days[di - 1]
        week_df.loc[m, "manual"] = True

man_un = st.session_state.get("manual_unassign", [])
if man_un:
    m_un = week_df["well"].isin(man_un)
    week_df.loc[m_un, "scheduled"] = False
    week_df.loc[m_un, "plan_unit"] = None
    week_df.loc[m_un, "day_idx"] = 0
    week_df.loc[m_un, "plan_day"] = pd.NaT
    week_df.loc[m_un, "manual"] = False

scheduled_all = week_df[week_df["scheduled"]].copy()

# --- BENTENG PERTAHANAN (SAFEGUARD) ---
# Memaksa Pandas membuat kolom jika secara gaib hilang dari memori saat kosong
if "timing" not in scheduled_all.columns:
    scheduled_all["timing"] = None
    scheduled_all["timing_label"] = None
    scheduled_all["out_dir"] = None
# --------------------------------------

_pd = scheduled_all["plan_day"] if len(scheduled_all) else pd.Series(dtype='datetime64[ns]')
_mn = scheduled_all["min_date"] if len(scheduled_all) else pd.Series(dtype='datetime64[ns]')
_mx = scheduled_all["max_date"] if len(scheduled_all) else pd.Series(dtype='datetime64[ns]')
_tipe = scheduled_all["tipe"] if len(scheduled_all) else pd.Series(dtype=object)
_rtag = scheduled_all.get("req_tag", pd.Series("", index=scheduled_all.index)).fillna("") if len(scheduled_all) else pd.Series(dtype=object)

_en = (_mn - _pd).dt.days if len(scheduled_all) else pd.Series(dtype=float)
_ln = (_pd - _mx).dt.days if len(scheduled_all) else pd.Series(dtype=float)
_oe = _pd < _mn if len(scheduled_all) else pd.Series(dtype=bool)
_ol = _pd > _mx if len(scheduled_all) else pd.Series(dtype=bool)

if len(scheduled_all):
    _is_in_range = (scheduled_all["min_date"] <= batch_hi) & (scheduled_all["max_date"] >= batch_lo)
else:
    _is_in_range = pd.Series(dtype=bool)

def _cat_lab(tipe, oe, ol, en, ln, tag, in_rng):
    if tipe in ["NW", "AWS"]: return "on-time", f"{tipe} (Prioritas)", ""
    in_window = not oe and not ol
    n = int(en) if oe else int(ln)
    arah = "early" if oe else "late"
    
    if tag in ["PRQ", "ORQ"]:
        if in_window: return "on-time", f"{tag} (on-time)", ""
        return tag, f"{tag} ({arah} {n} hari)", arah
    
    if in_window: return "on-time", "", ""
    if not in_rng: return "on-time", f"Out of Window ({arah} {n}d)", ""
        
    return ("EARLY" if oe else "LATE"), f"{arah} {n} hari", arah

if len(scheduled_all) > 0:
    _cats = [_cat_lab(tp, oe, ol, en, ln, tg, rng) for tp, oe, ol, en, ln, tg, rng in zip(_tipe, _oe, _ol, _en.fillna(0), _ln.fillna(0), _rtag, _is_in_range)]
    scheduled_all["timing"] = [c[0] for c in _cats]
    scheduled_all["timing_label"] = [c[1] for c in _cats]
    scheduled_all["out_dir"] = [c[2] for c in _cats]
else:
    scheduled_all["timing"] = None
    scheduled_all["timing_label"] = None
    scheduled_all["out_dir"] = None

sched_wells = set(scheduled_all["well"]) if len(scheduled_all) else set()
leftover = week_df[~week_df["scheduled"]].copy()
missed = leftover[leftover["max_date"] <= batch_hi] if len(leftover) else leftover.copy()

# ── Render Header & KPIs via WELLGO UI ──────────────────────────────────────
total_scheduled = len(scheduled_all)
total_missed_dl = len(missed)
total_eligible = len(elig_all)

total_kpi_target = total_scheduled + total_missed_dl
comp_rate = int(100 * total_scheduled / total_kpi_target) if total_kpi_target > 0 else 100

kr_calc = []
total_minutes = 0
for (di, dday, unit), sub in scheduled_all.groupby(["day_idx", "plan_day", "plan_unit"]):
    c = sub[sub["has_coord"]]
    dist_val = route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0
    kr_calc.append(dist_val)
    dur_sum = int(pd.to_numeric(sub["dur"], errors="coerce").fillna(0).sum()) if "dur" in sub.columns else 0
    total_minutes += dur_sum + (dist_val / max(float(speed), 1.0)) * 60

computed_total_km = sum(kr_calc)
avg_utilization = (total_minutes / (max(len(scheduled_all["plan_unit"].unique()), 1) * horizon * time_budget)) * 100 if time_budget > 0 else 0

ui.hero_header(
    date_str=plan_start_ts.strftime("%d %b %Y"), 
    horizon=horizon, 
    units=len(scheduled_all["plan_unit"].unique()) if len(scheduled_all) else 0, 
    compliance=comp_rate, 
    mode=mode
)

ui.kpi_row([
    ("wells scheduled", f"{total_scheduled}", f"/{total_eligible}", ui.TEAL_GREEN),
    ("miss deadline",   f"{total_missed_dl}", " wells", ui.RED),
    ("wells off",       f"{len(off_wells)}", " wells", "#64748B"),
    ("total route",     f"{computed_total_km:.0f}", " km", ui.TEAL),
    ("avg utilization", f"{avg_utilization:.0f}", "%",  ui.AMBER),
])

# ── Main Workspace Tabs ────────────────────────────────────────────────────
tab_guide, tab_sched, tab_map, tab_matrix, tab_cart, tab_sch, tab_diagnostics = st.tabs([
    "📘 Panduan",
    "📅 Jadwal Operasional",
    "🗺️ Peta Rute", 
    "📊 Matriks Deviasi", 
    "🛒 Cart Manual",
    "🗃️ SCH Database",
    "🔎 Diagnostik"
])

with tab_guide:
    guide.render_guide()

with tab_sched:
    if len(scheduled_all) == 0:
        st.info("Belum ada jadwal yang berhasil dialokasikan pada siklus ini.")
    else:
        for day_idx, day_date in enumerate(days, 1):
            day_data = scheduled_all[scheduled_all["day_idx"] == day_idx]
            if len(day_data) == 0: continue
            
            ui.day_header(f"Hari Ke-{day_idx}", day_date.strftime("%A, %d %b"), 
                          units=day_data["plan_unit"].nunique(), wells=len(day_data))
            
            with st.expander(f"⚙️ Atur Manual Sumur Hari Ke-{day_idx}"):
                ca1, ca2 = st.columns(2)
                with ca1:
                    ui.section("➖ Keluarkan Sumur", eyebrow="Batal jadwalkan dari hari ini")
                    to_rm = st.multiselect("Pilih sumur:", day_data["well"].tolist(), key=f"rm_w_{day_idx}")
                    if st.button("Keluarkan", key=f"btn_rm_{day_idx}", use_container_width=True):
                        st.session_state.setdefault("manual_unassign", [])
                        for w in to_rm:
                            if w in st.session_state.get("manual_assign", {}):
                                del st.session_state["manual_assign"][w]
                            if w not in st.session_state["manual_unassign"]:
                                st.session_state["manual_unassign"].append(w)
                        st.rerun()
                with ca2:
                    ui.section("➕ Tambahkan Sumur", eyebrow="Cari dan masukkan sumur")
                    avail = leftover["well"].tolist()
                    to_add = st.multiselect("Cari sumur (ketik nama):", sorted(avail), key=f"add_w_{day_idx}")
                    target_u = st.selectbox("Pilih Unit:", ALL_UNITS, key=f"add_u_{day_idx}")
                    if st.button("Tambahkan ke Unit", key=f"btn_add_{day_idx}", type="primary", use_container_width=True):
                        st.session_state.setdefault("manual_assign", {})
                        st.session_state.setdefault("manual_unassign", [])
                        for w in to_add:
                            st.session_state["manual_assign"][w] = {"unit": target_u, "day_idx": day_idx}
                            if w in st.session_state["manual_unassign"]:
                                st.session_state["manual_unassign"].remove(w)
                        st.rerun()
            
            remote_data = day_data[day_data["zone"] == "remote"]
            nonremote_data = day_data[day_data["zone"] == "non-remote"]
            
            if not remote_data.empty:
                st.markdown("<div style='font-size:14px; font-weight:700; color:#5E7076; margin: 20px 0 10px 0; padding-bottom:5px; border-bottom:1px solid #DCE4E6;'>📍 KELOMPOK UNIT REMOTE (BANGKO / BALAM)</div>", unsafe_allow_html=True)
                for unit, sub in remote_data.groupby("plan_unit"):
                    c = sub[sub["has_coord"]] if "has_coord" in sub.columns else sub
                    dist = route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0
                    dur_sum = int(pd.to_numeric(sub["dur"], errors="coerce").fillna(0).sum()) if "dur" in sub.columns else 0
                    est_min = dur_sum + (dist / max(float(speed), 1.0)) * 60
                    pct = (est_min / time_budget) * 100 if time_budget > 0 else 0
                    subarea = ", ".join(sorted(sub["subarea"].dropna().astype(str).unique())) if "subarea" in sub.columns else ""
                    
                    wells_list = []
                    for _, w in sub.iterrows():
                        tipe = w.get("tipe", "")
                        rtag = w.get("req_tag", "")
                        
                        if tipe == "NW": cat = "NW"
                        elif tipe == "AWS": cat = "AWS"
                        elif rtag == "PRQ": cat = "PRQ"
                        elif rtag == "ORQ": cat = "ORQ"
                        else: cat = "RTN"
                        
                        min_d = w["min_date"].strftime("%d/%m") if pd.notna(w["min_date"]) else "-"
                        max_d = w["max_date"].strftime("%d/%m") if pd.notna(w["max_date"]) else "-"
                        dur_val = int(w["dur"]) if pd.notna(w["dur"]) else 0
                        
                        st_type = str(w.get("string_type", "")).strip().upper()
                        is_gas = st_type == "GP"
                        well_disp = f"{w['well']} <b style='color:#E6B23A;font-size:10px;'>[GAS]</b>" if is_gas else w["well"]
                        
                        wells_list.append((well_disp, cat, f"{min_d} ➔ {max_d}", f"{dur_val}m"))
                    
                    ui.unit_card(unit, subarea, km=dist, minutes=est_min, pct=pct, wells=wells_list)

            if not nonremote_data.empty:
                st.markdown("<div style='font-size:14px; font-weight:700; color:#5E7076; margin: 20px 0 10px 0; padding-bottom:5px; border-bottom:1px solid #DCE4E6;'>📍 KELOMPOK UNIT NON-REMOTE (BEKASAP)</div>", unsafe_allow_html=True)
                for unit, sub in nonremote_data.groupby("plan_unit"):
                    c = sub[sub["has_coord"]] if "has_coord" in sub.columns else sub
                    dist = route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0
                    dur_sum = int(pd.to_numeric(sub["dur"], errors="coerce").fillna(0).sum()) if "dur" in sub.columns else 0
                    est_min = dur_sum + (dist / max(float(speed), 1.0)) * 60
                    pct = (est_min / time_budget) * 100 if time_budget > 0 else 0
                    subarea = ", ".join(sorted(sub["subarea"].dropna().astype(str).unique())) if "subarea" in sub.columns else ""
                    
                    wells_list = []
                    for _, w in sub.iterrows():
                        tipe = w.get("tipe", "")
                        rtag = w.get("req_tag", "")
                        
                        if tipe == "NW": cat = "NW"
                        elif tipe == "AWS": cat = "AWS"
                        elif rtag == "PRQ": cat = "PRQ"
                        elif rtag == "ORQ": cat = "ORQ"
                        else: cat = "RTN"
                        
                        min_d = w["min_date"].strftime("%d/%m") if pd.notna(w["min_date"]) else "-"
                        max_d = w["max_date"].strftime("%d/%m") if pd.notna(w["max_date"]) else "-"
                        dur_val = int(w["dur"]) if pd.notna(w["dur"]) else 0
                        
                        st_type = str(w.get("string_type", "")).strip().upper()
                        is_gas = st_type == "GP"
                        well_disp = f"{w['well']} <b style='color:#E6B23A;font-size:10px;'>[GAS]</b>" if is_gas else w["well"]
                        
                        wells_list.append((well_disp, cat, f"{min_d} ➔ {max_d}", f"{dur_val}m"))
                    
                    ui.unit_card(unit, subarea, km=dist, minutes=est_min, pct=pct, wells=wells_list)
            
with tab_map:
    day_labels = [days[i].strftime("%Y-%m-%d") for i in range(horizon)]
    lbl2idx = {lbl: i + 1 for i, lbl in enumerate(day_labels)}

    c_flt1, c_flt2 = st.columns([3, 1])
    with c_flt1:
        sel_labels = st.multiselect("🗓️ Fokus Tanggal Rute (Pilih 1 untuk view harian aktif)", day_labels, default=day_labels)
        if not sel_labels: sel_labels = day_labels
    with c_flt2:
        dur_pick = st.multiselect("⏱️ Filter Durasi Test", [30, 60], default=[30, 60])
        if not dur_pick: dur_pick = [30, 60]

    sel_idx = sorted(lbl2idx[l] for l in sel_labels)
    is_single_day = len(sel_idx) == 1
    view_day = days[sel_idx[0] - 1] if (is_single_day and len(sel_idx) > 0) else None

    disp = scheduled_all[scheduled_all["day_idx"].isin(sel_idx) & scheduled_all["dur"].isin(dur_pick)].copy() if len(scheduled_all) else scheduled_all.copy()
    
    mco1, mco2, mco3 = st.columns([1.5, 2, 1.4])
    color_mode = mco1.selectbox("🎨 Skema Pewarnaan Peta", ["Otomatis (hari/unit)", "Zona remote/non-remote", "Per unit", "Early / Late test"])
    unit_filter = mco2.multiselect("🔧 Batasi Tampilan Unit MWT", sorted(scheduled_all["plan_unit"].dropna().unique().tolist()) if len(scheduled_all) else [])
    search_q = mco3.text_input("🔎 Pencarian Cepat Nama Sumur", placeholder="Contoh: BO083").strip().upper()
    timing_pick = st.multiselect("🕐 Filter Deviasi Window", ["EARLY", "on-time", "LATE", "PRQ", "ORQ"], default=[])
    field_block = st.multiselect("📦 Tampilkan Batas Field Area", field_list, default=[])
    
    fb_wells = field_wells_coord[field_wells_coord["field"].isin(field_block)] if field_block else field_wells_coord.iloc[0:0]

    pmap = disp[disp["has_coord"]].copy() if len(disp) else disp.copy()
    if unit_filter: pmap = pmap[pmap["plan_unit"].isin(unit_filter)]
    if timing_pick: pmap = pmap[pmap["timing"].isin(timing_pick)]
    
    man_pick = []
    prev = leftover[leftover["well"].isin(man_pick) & leftover["has_coord"]].copy() if man_pick and len(leftover) else leftover.iloc[0:0]
    search_terms = [t for t in search_q.replace(",", " ").split() if t]
    search_hits = leftover.iloc[0:0]
    
    if search_terms and len(week_df):
        _src = week_df[week_df["has_coord"]].copy()
        _wu = _src["well"].str.upper()
        smask = pd.Series(False, index=_src.index)
        for t in search_terms: smask |= _wu.str.contains(t, regex=False)
        search_hits = _src[smask].copy()
        if len(search_hits):
            sh = search_hits.assign(
                tgl=search_hits["plan_day"].apply(lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "—"),
                grup=search_hits["plan_unit"].fillna("belum terjadwal"))
            info = "; ".join(f"**{r.well}** → {r.grup} ({r.tgl})" for r in sh.itertuples())
            st.info(f"🔎 Hasil Pencarian Spasial: {info}")

    if len(pmap) or len(prev) or len(search_hits) or field_block:
        TIPE_RING = {"NW": [220, 30, 30], "AWS": [245, 150, 20], "REG": [120, 120, 120]}
        TIMING_COL = {"EARLY": [30, 120, 220], "on-time": [150, 150, 150], "LATE": [220, 30, 30], "PRQ": [150, 80, 200], "ORQ": [0, 160, 140]}
        legend = ""
        
        if len(pmap):
            if color_mode == "Zona remote/non-remote":
                ZCOL = {"remote": [30, 120, 220], "non-remote": [240, 140, 30]}
                pmap["color"] = pmap["zone"].map(lambda z: ZCOL.get(z, [130, 130, 130]))
                legend = "🔵 Remote (Bangko/Balam) · 🟠 Non-Remote (Bekasap)"
            elif color_mode == "Per unit":
                ulabels = sorted(pmap["plan_unit"].dropna().unique())
                pmap["color"] = pmap["plan_unit"].apply(lambda k: cmap(k, ulabels))
                legend = "Skala Warna Berdasarkan Distribusi ID Unit"
            elif color_mode == "Early / Late test":
                pmap["color"] = pmap["timing"].map(lambda t: TIMING_COL.get(t, [150, 150, 150]))
                legend = "🔵 EARLY · ⚪ ON-TIME · 🔴 LATE · 🟣 PRQ · 🟢 ORQ"
            else:
                if is_single_day:
                    labels = sorted(pmap["plan_unit"].dropna().unique()); pmap["ckey"] = pmap["plan_unit"]
                else:
                    labels = sorted(pmap["day_idx"].unique()); pmap["ckey"] = pmap["day_idx"]
                pmap["color"] = pmap["ckey"].apply(lambda k: cmap(k, labels))
                legend = "Dimensi Warna: Skema Penjadwalan Kalender Hari Operasional"
            
            pmap["radius"] = np.where(pmap["coord_source"].str.startswith("imputed"), 90, 170)
            pmap["radius"] = pmap["radius"] * np.where(pmap["dur"] == 30, 0.7, 1.0)
            pmap["hit"] = pmap["well"].str.upper().isin(search_terms) if search_terms else False
            pmap["ring"] = pmap.apply(lambda r: [255, 235, 0] if r["hit"] else TIPE_RING.get(r["tipe"], [120, 120, 120]), axis=1)
            pmap["ringw"] = np.where(pmap["hit"], 6, np.where(pmap["tipe"].isin(["NW", "AWS"]), 3, 0))

        def _tipcols(d):
            if not len(d): return d
            d["tgl_str"] = d["plan_day"].dt.strftime("%Y-%m-%d").fillna("belum terjadwal")
            d["min_str"] = d["min_date"].dt.strftime("%Y-%m-%d").fillna("—")
            d["max_str"] = d["max_date"].dt.strftime("%Y-%m-%d").fillna("—")
            if "timing_label" not in d.columns: d["timing_label"] = ""
            d["ket"] = d["timing_label"].replace("", "-")
            return d
            
        pmap = _tipcols(pmap)
        layers = []
        
        if field_block:
            FCOL = [[120, 80, 200], [0, 150, 136], [200, 100, 0], [60, 130, 200]]
            for fi, fld in enumerate(field_block):
                fw = field_wells_coord[field_wells_coord["field"] == fld]
                if not len(fw): continue
                col = FCOL[fi % len(FCOL)]
                if len(fw) >= 3:
                    layers.append(pdk.Layer("PolygonLayer", data=[{"polygon": block_polygon(fw), "color": col + [50]}],
                        get_polygon="polygon", get_fill_color="color", get_line_color=col, line_width_min_pixels=2, stroked=True, filled=True))
                fwp = fw.copy(); fwp["fld"] = fld; fwp["fcol"] = [col] * len(fwp)
                layers.append(pdk.Layer("ScatterplotLayer", data=fwp, get_position=["lon", "lat"], get_fill_color="fcol", get_radius=110, opacity=0.55))
        
        draw_units = len(pmap) and (is_single_day or color_mode == "Per unit")
        if draw_units:
            if show_block:
                polys = [{"polygon": block_polygon(sub), "color": list(sub["color"].iloc[0]) + [55]} for u, sub in pmap.groupby("plan_unit") if len(sub) >= 3]
                if polys: layers.append(pdk.Layer("PolygonLayer", data=polys, get_polygon="polygon", get_fill_color="color", get_line_color="color", line_width_min_pixels=1, stroked=True, filled=True))
            lines = []
            for u, sub in pmap.groupby("plan_unit"):
                s = sub.reset_index(drop=True)
                order, _ = optimize_route(s["lat"].values, s["lon"].values)
                col = list(s["color"].iloc[0])
                for a in range(len(order) - 1):
                    i, j = order[a], order[a + 1]
                    lines.append({"from": [s.loc[i, "lon"], s.loc[i, "lat"]], "to": [s.loc[j, "lon"], s.loc[j, "lat"]], "color": col})
            if lines: layers.append(pdk.Layer("LineLayer", data=pd.DataFrame(lines), get_source_position="from", get_target_position="to", get_color="color", get_width=2))
        
        if len(pmap):
            layers.append(pdk.Layer("ScatterplotLayer", data=pmap, get_position=["lon", "lat"], get_fill_color="color", get_radius="radius", get_line_color="ring", get_line_width="ringw", line_width_min_pixels=1, stroked=True, filled=True, pickable=True, opacity=0.9))
        
        if len(search_hits):
            foc = search_hits
            zoom_lvl = 13.5 if len(search_hits) == 1 else 11.0
            
            layers.append(pdk.Layer(
                "TextLayer",
                data=search_hits.copy(),
                get_position=["lon", "lat"],
                get_text="well",
                get_size=75,
                get_color=[255, 235, 0],
                get_pixel_offset=[0, -45],
                font_family="Inter",
                font_weight="bold",
                pickable=False
            ))
            
        elif len(pmap):
            foc = pmap
            zoom_lvl = 8.5
        elif field_block and len(fb_wells):
            foc = fb_wells
            zoom_lvl = 10.0
        else:
            foc = leftover.iloc[0:0]
            zoom_lvl = 8.5
            
        lat_init = float(foc["lat"].mean()) if len(foc) else 1.6
        lon_init = float(foc["lon"].mean()) if len(foc) else 101.3
        
        view = pdk.ViewState(latitude=lat_init, longitude=lon_init, zoom=zoom_lvl)
        tip = "{well} [{tipe}] · {ket}\nTanggal Plan: {tgl_str} | Unit: {plan_unit}\nWindow Execution: {min_str} → {max_str}"
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="road", tooltip={"text": tip}))
        st.caption(f"💡 {legend}. Ring Merah=NW, Oranye=AWS. Garis biru menghubungkan sequence rute TSP antar sumur.")

with tab_matrix:
    ui.section("Matriks Deviasi Jadwal", eyebrow="Evaluasi kepatuhan min-max date")
    sa = scheduled_all
    
    if len(sa) > 0:
        def _wn(mask): return len(sa.loc[mask])
            
        tim, dr = sa["timing"], sa["out_dir"]
        e_pure = _wn(tim == "EARLY")
        e_req = _wn(tim.isin(["PRQ", "ORQ"]) & (dr == "early"))
        l_pure = _wn(tim == "LATE")
        l_req = _wn(tim.isin(["PRQ", "ORQ"]) & (dr == "late"))
        
        matrix_data = [
            {"Kategori Deviasi Operasional": "Murni (Rentang Pengetesan)", "⏪ Total EARLY": e_pure, "⏩ Total LATE": l_pure},
            {"Kategori Deviasi Operasional": "PRQ / ORQ (Request Ops/PE)", "⏪ Total EARLY": e_req, "⏩ Total LATE": l_req}
        ]
        df_matrix = pd.DataFrame(matrix_data)
        df_matrix.loc[len(df_matrix)] = ["TOTAL KESELURUHAN DEVIASI", e_pure + e_req, l_pure + l_req]
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)
        
        ui.section("Rincian Evaluasi Window", eyebrow="Tabel kontrol compliance")
        if (tim != "on-time").any():
            detail_cols = ["well", "plan_unit", "timing", "timing_label", "plan_day", "min_date", "max_date", "next_wt"]
            df_detail = sa.loc[tim != "on-time", detail_cols].copy()
            df_detail["plan_day"] = df_detail["plan_day"].dt.strftime("%Y-%m-%d")
            df_detail["min_date"] = df_detail["min_date"].dt.strftime("%Y-%m-%d").fillna("-")
            df_detail["max_date"] = df_detail["max_date"].dt.strftime("%Y-%m-%d").fillna("-")
            df_detail["next_wt"] = df_detail["next_wt"].dt.strftime("%Y-%m-%d").fillna("-") if "next_wt" in df_detail.columns else "-"
            
            df_detail = df_detail.rename(columns={
                "well": "Well", "plan_unit": "Unit Assigned", "timing": "Kategori",
                "timing_label": "Deviasi Analisis", "plan_day": "Tanggal Sched",
                "min_date": "Earliest Date", "max_date": "Latest Date", "next_wt": "Next Proposed WT"
            })
            st.dataframe(df_detail.sort_values(["Kategori", "Tanggal Sched"]), use_container_width=True, hide_index=True)
        else:
            st.success("✨ Sempurna! Seluruh aset sumur tereksekusi On-Time di dalam rentang window fisis.")
    else:
        st.info("Belum ada data perencanaan mingguan untuk dianalisis.")
    
    if len(missed):
        ui.section("Daftar Miss Deadline (Kapasitas Penuh)", eyebrow="Butuh aksi manual/tambah shift")
        st.dataframe(missed[["well", "unit", "subarea", "category", "urgency", "max_date"]].rename(columns={"max_date": "deadline", "unit": "unit_asli"}).sort_values("urgency"), use_container_width=True, hide_index=True)

    with st.expander(f"🔍 Evaluasi Pengecualian Kandidat (Ter-Skip) - Klik Untuk Expand"):
        elig_set = set(elig_all["well"])
        out = raw[~raw["well"].isin(elig_set)].copy()
        no_date = out[out["min_date"].isna() | out["max_date"].isna()]
        is_comp = out[out["well"].isin(executed)]
        is_off = out[out["status"] == "OFF"]
        is_woff = out[out["well"].isin(woff_set)]
        nw_out = out[out["is_nwaws"].fillna(False) & ~((out["min_date"] <= batch_hi) & (out["max_date"] >= batch_lo))]
        accounted = (set(no_date["well"]) | set(is_comp["well"]) | set(is_off["well"]) | set(is_woff["well"]) | set(nw_out["well"]))
        win_out = out[~out["well"].isin(accounted) & out["min_date"].notna() & out["max_date"].notna()]
        st.markdown(
            f"- **Formula Excel Kosong (Min/Max Date)**: {len(no_date)} sumur dibuang karena window tak terbaca.\n"
            f"- **Diluar Rentang Siklus**: {len(win_out)} sumur due di luar horizon. Lebarkan periode jika ingin disertakan.\n"
            f"- **NW/AWS Diluar Siklus**: {len(nw_out)} sumur.\n"
            f"- **Status Exclude**: {len(is_comp)} COMP, {len(is_off)} OFF, {len(is_woff)} NCMP-WOFF.")

with tab_cart:
    ui.section("Matriks Ketersediaan Kapasitas", eyebrow="Visualisasi load per unit harian")
    grid_cols = [f"Hari {d}" for d in range(1, horizon + 1)]
    grid_df = pd.DataFrame(index=ALL_UNITS, columns=grid_cols, data="")
    for u in ALL_UNITS:
        for d in range(1, horizon + 1):
            cnt = int(((scheduled_all["plan_unit"] == u) & (scheduled_all["day_idx"] == d)).sum()) if len(scheduled_all) else 0
            if cnt == 0: val = "kosong"
            elif cnt < max_wells: val = f"🟢 {cnt}/{max_wells}"
            elif cnt == max_wells: val = f"✅ {cnt}/{max_wells}"
            else: val = f"⚠️ {cnt}/{max_wells} (Over)"
            grid_df.at[u, f"Hari {d}"] = val
    st.dataframe(grid_df, use_container_width=True)

    ui.section("Smart Cart Assistant", eyebrow="Assign manual berdasar rekomendasi kedekatan")
    recs = []
    if len(scheduled_all) > 0 and len(leftover) > 0:
        for _, w in leftover.iterrows():
            if not w['has_coord']: continue
            valid_sched = scheduled_all[scheduled_all['plan_day'] <= w['max_date']] if pd.notna(w['max_date']) else scheduled_all
            if valid_sched.empty: valid_sched = scheduled_all
            if valid_sched.empty: continue

            dists = haversine_km(w['lat'], w['lon'], valid_sched['lat'].values, valid_sched['lon'].values)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            best_match = valid_sched.iloc[best_idx]

            target_u = best_match['plan_unit']
            target_d = int(best_match['day_idx'])
            curr_cnt = int(((scheduled_all["plan_unit"] == target_u) & (scheduled_all["day_idx"] == target_d)).sum())
            basket_str = f"{curr_cnt}/{max_wells}" + (" ⚠️ (Penuh)" if curr_cnt >= max_wells else "")

            recs.append({
                "Pilih": False, "Well": w['well'], "Deadline": w['max_date'].strftime('%Y-%m-%d') if pd.notna(w['max_date']) else '-',
                "Target Unit": target_u, "Hari ke-": target_d, "Isi Keranjang": basket_str, "Jarak Kedekatan (km)": round(best_dist, 1),
                "Status": "⚠️ Miss Deadline" if w['well'] in missed['well'].values else "Sisa Pool"
            })

    if recs:
        rec_df = pd.DataFrame(recs)
        f1, f2, f3 = st.columns([2, 2, 1.5])
        flt_unit = f1.selectbox("Filter Unit Armada:", ["Semua Unit"] + ALL_UNITS, key="cart_u")
        flt_day = f2.selectbox("Filter Hari Kerja Horizon:", ["Semua Hari"] + list(range(1, horizon + 1)), key="cart_d")
        flt_miss = f3.checkbox("Tampilkan Hanya Item Miss Deadline", value=True, key="cart_m")

        view_df = rec_df.copy()
        if flt_unit != "Semua Unit": view_df = view_df[view_df["Target Unit"] == flt_unit]
        if flt_day != "Semua Hari": view_df = view_df[view_df["Hari ke-"] == int(flt_day)]
        if flt_miss: view_df = view_df[view_df["Status"].str.contains("Miss Deadline")]

        view_df = view_df.sort_values(["Hari ke-", "Target Unit", "Jarak Kedekatan (km)"])
        
        edited_rec = st.data_editor(
            view_df, hide_index=True, use_container_width=True,
            column_config={
                "Pilih": st.column_config.CheckboxColumn("Masukin Armada?", default=False),
                "Target Unit": st.column_config.SelectboxColumn("Ubah Unit Logistik", options=ALL_UNITS),
                "Hari ke-": st.column_config.NumberColumn("Ubah Hari Horizon", min_value=1, max_value=horizon)
            },
            disabled=["Well", "Deadline", "Isi Keranjang", "Jarak Kedekatan (km)", "Status"]
        )

        if st.button("🪄 Validasi & Masukkan ke Keranjang MWT", type="primary"):
            selected_recs = edited_rec[edited_rec["Pilih"] == True]
            if not selected_recs.empty:
                st.session_state.setdefault("manual_assign", {})
                for _, row in selected_recs.iterrows():
                    st.session_state["manual_assign"][row["Well"]] = {"unit": row["Target Unit"], "day_idx": row["Hari ke-"]}
                st.rerun()
    else:
        st.info("Tidak ada sisa sumur yang membutuhkan assign rekomendasi.")
        
    with st.expander("🛠️ Bypass Override: Assign Manual Buta Tanpa Jarak"):
        left_opts = sorted(leftover["well"].tolist())
        miss_opts = sorted(missed["well"].tolist())
        mc1, mc2 = st.columns([3, 2])
        with mc1:
            man_pick_all = st.multiselect("Pilih Sumur Terbuang", left_opts, key="man_pick")
            man_pick_miss = st.multiselect(f"Miss deadline krisis ({len(miss_opts)})", miss_opts, key="man_pick_miss")
        man_pick = sorted(set(man_pick_all) | set(man_pick_miss))
        with mc2:
            a1, a2 = st.columns(2)
            man_unit = a1.selectbox("Pilih Unit", ALL_UNITS, key="man_unit")
            man_day = a2.selectbox("Hari ke-", list(range(1, horizon + 1)), key="man_day")
            if st.button("➕ Force Assign", use_container_width=True, disabled=not man_pick):
                st.session_state.setdefault("manual_assign", {})
                for w in man_pick:
                    st.session_state["manual_assign"][w] = {"unit": man_unit, "day_idx": int(man_day)}
                st.rerun()
    
    if man:
        st.write("**Histori Assign Manual Teraktivasi:**")
        for w, info in list(man.items()):
            r1, r2 = st.columns([5, 1])
            warn = ""
            cnt = int(((scheduled_all["plan_unit"] == info["unit"]) & (scheduled_all["day_idx"] == info["day_idx"])).sum())
            if cnt > max_wells: warn = f" ⚠️ (Memicu Overload: {cnt} well)"
            r1.write(f"• **{w}** → {info['unit']} (Hari {info['day_idx']}){warn}")
            if r2.button("Hapus", key=f"rm_{w}"):
                del st.session_state["manual_assign"][w]
                st.rerun()

with tab_sch:
    ui.section("Dashboard Status Realisasi", eyebrow=f"Periode {per_lo} s/d {per_hi}")
    
    tot_comp = len(comp_wells)
    tot_ncmp = len(ncmp_df[ncmp_df["well"].isin(ncmp_set)])
    tot_replan = len(replan_df)
    tot_woff = len(woff_wells)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>Total COMP</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:{ui.TEAL_GREEN};'>{tot_comp}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>Total NCMP</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:#E67E22;'>{tot_ncmp}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>NCMP (Dijadwal Ulang)</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:{ui.TEAL};'>{tot_replan}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>NCMP (Sumur OFF / Skip)</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:{ui.RED};'>{tot_woff}</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["✅ Data COMP", "🔁 NCMP (Dijadwalkan Ulang)", "⏸️ NCMP (Skip / OFF)"])
    with t1:
        if len(comp_wells):
            st.dataframe(comp_wells[["well", "unit", "subarea", "category", "dur", "sch_status"]].rename(columns={"unit": "unit_asli", "dur": "durasi", "sch_status": "SCH"}), use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada sumur COMP di periode ini.")
    with t2:
        if len(replan_df):
            st.dataframe(replan_df.rename(columns={"plan_date": "tgl_NCMP", "reason": "alasan"}), use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada sumur NCMP yang dijadwalkan ulang.")
    with t3:
        if len(woff_wells):
            st.dataframe(woff_wells[["well", "unit", "subarea", "category", "max_date", "status"]].rename(columns={"unit": "unit_asli", "max_date": "deadline"}), use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada sumur OFF yang di-skip.")
            
    st.divider()
    ui.section("Data Mentah SCH Database", eyebrow="Informasi dari file yang diunggah")
    if comp_files:
        for f in comp_files:
            st.markdown(f"**File:** `{f.name}`")
            try:
                df_raw = pd.read_excel(BytesIO(f.getvalue()))
                st.dataframe(df_raw, use_container_width=True)
            except Exception as e:
                st.error(f"Gagal memuat pratinjau untuk file ini: {str(e)}")
    else:
        st.info("Belum ada file COMP/NCMP yang diunggah pada menu 'Status Realisasi Harian' di sidebar.")

with tab_diagnostics:
    ui.section("Analisis Akumulasi Jarak Tempuh", eyebrow="Agregasi pergerakan armada fisik")
    
    kr_analysis = []
    for (di, dday, unit), sub in scheduled_all.groupby(["day_idx", "plan_day", "plan_unit"]):
        c = sub[sub["has_coord"]]
        dist_val = route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0
        kr_analysis.append({"Hari": int(di), "Tanggal": dday.date(), "Unit": unit, "km": round(float(dist_val), 1), "Sumur": len(sub)})
    
    if kr_analysis:
        kdf_an = pd.DataFrame(kr_analysis)
        piv_an = kdf_an.pivot_table(index="Unit", columns="Hari", values="km", aggfunc="sum", fill_value=0.0)
        piv_an.columns = [days[c - 1].strftime("%Y-%m-%d") for c in piv_an.columns]
        piv_an["Total Jarak (km)"] = piv_an.sum(axis=1)
        st.dataframe(piv_an.round(1), use_container_width=True)
        
        ui.section("Tren Jarak Geografis Harian", eyebrow="Fluktuasi KM route")
        per_day = kdf_an.groupby(["Tanggal"]).agg(km=("km", "sum"), Unit=("Unit", "nunique"), Sumur=("Sumur", "sum")).reset_index()
        per_day["km/sumur"] = (per_day["km"] / per_day["Sumur"].clip(lower=1)).round(2)
        per_day["Tanggal"] = per_day["Tanggal"].astype(str)
        st.bar_chart(per_day.set_index("Tanggal")["km"])
    else:
        st.info("Unggah berkas untuk melihat visualisasi matriks rute.")
        
    ui.section("Raw Data Perencanaan Rute", eyebrow="Tabel breakdown logistik operasional")
    if len(disp):
        scols_view = ["day_idx", "plan_day", "plan_unit", "manual", "timing_label", "tipe", "well", "subarea", "dur", "min_date", "max_date"]
        det_view = disp[scols_view].rename(columns={"day_idx": "Hari", "plan_day": "Tanggal", "plan_unit": "MWT Group", "timing_label": "Analisis Window", "dur": "Durasi (Min)"})
        det_view["Tanggal"] = det_view["Tanggal"].dt.strftime("%Y-%m-%d")
        det_view["min_date"] = det_view["min_date"].dt.strftime("%Y-%m-%d")
        det_view["max_date"] = det_view["max_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(det_view.sort_values(["Hari", "MWT Group"]), use_container_width=True, hide_index=True)
    
    st.divider()
    ui.section("Modul Sinkronisasi Eksekusi & Export", eyebrow="Konfirmasi realisasi harian")
    if view_day is not None:
        st.caption(f"Tandai sumur yang telah dirampungkan secara fisik di lapangan pada **{view_day.date()}**. Ini akan mengecualikan sumur tersebut dari siklus rollout berikutnya.")
        done = st.multiselect("Pilih Sumur Terealisasi", sorted(disp["well"]))
        if st.button("💾 Simpan Status Eksekusi", type="primary"):
            rows = [(r["well"], str(r["plan_unit"]), "executed" if r["well"] in done else "planned") for _, r in disp.iterrows()]
            save_status(str(view_day.date()), rows)
            st.success(f"Log tersimpan. Lakukan refresh aplikasi untuk komputasi ulang rute sisa.")
    
    XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    exp_cols = ["day_idx", "plan_day", "plan_unit", "manual", "timing", "timing_label", "tipe", "zone", "unit", "well", "subarea", "field", "category", "dur", "min_date", "max_date", "urgency", "coord_source", "lat", "lon"]
    ren = {"day_idx": "hari", "plan_day": "tanggal", "plan_unit": "grup", "manual": "manual", "timing_label": "early_late", "unit": "unit_asli", "dur": "durasi_test_menit", "max_date": "deadline"}

    ex1, ex2 = st.columns(2)
    out_w = BytesIO()
    with pd.ExcelWriter(out_w, engine="openpyxl") as w:
        scheduled_all[exp_cols].rename(columns=ren).sort_values(["hari", "grup", "urgency"]).to_excel(w, sheet_name="Jadwal_Mingguan", index=False)
        if len(missed): missed[["well", "unit", "subarea", "category", "dur", "urgency", "max_date"]].rename(columns={"dur": "durasi_test_menit", "max_date": "deadline", "unit": "unit_asli"}).to_excel(w, sheet_name="Miss-Deadline", index=False)
        if len(off_wells): off_wells[["well", "unit", "subarea", "category", "dur", "status"]].rename(columns={"unit": "unit_asli", "dur": "durasi_test_menit"}).to_excel(w, sheet_name="Well-OFF", index=False)
    ex1.download_button("⬇️ Unduh Master Mingguan (.xlsx)", out_w.getvalue(), file_name=f"jadwal_mingguan_{week_lo.date()}_{week_hi.date()}.xlsx", mime=XLSX_MIME)

    if view_day is not None:
        out_d = BytesIO()
        with pd.ExcelWriter(out_d, engine="openpyxl") as w:
            disp[exp_cols].rename(columns=ren).sort_values(["grup", "urgency"]).to_excel(w, sheet_name="Jadwal_Harian", index=False)
            unit_summary(disp, speed).to_excel(w, sheet_name="Ringkasan_Unit", index=False)
        ex2.download_button(f"⬇️ Unduh Rute Harian {view_day.date()} (.xlsx)", out_d.getvalue(), file_name=f"jadwal_harian_{view_day.date()}.xlsx", mime=XLSX_MIME, type="primary")

# ── Footer Cleanup Action Trigger Module ────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns([4, 1])
with col_f1:
    st.caption("WELLGO (Well Grouping Optimizer) SL North. Hak Cipta Operasional 2026. Dikembangkan secara analitik dengan Python & Streamlit Engine.")
with col_f2:
    if st.button("🔄 Hard Reset Konfigurasi", use_container_width=True):
        st.session_state["manual_assign"] = {}
        st.session_state["manual_unassign"] = []
        st.session_state["field_assign"] = {}
        st.rerun()