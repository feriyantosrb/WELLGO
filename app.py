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

def reset_execution_log():
    """Kosongkan seluruh SCH_Database (execution_log): COMP/NCMP hasil upload + tanda COMP manual."""
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

    # status pemenang per well (tanggal terbaru): executed > ncmp > pending
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
    """COMP (executed) per well dari execution_log → {well: [(Timestamp, reason_upper), ...]} (seluruh log).
    Dipakai utk deteksi fase AWS: kolom Reason (AS1/AS2) sbg sinyal utama, window POP sbg fallback."""
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT well_name AS well, plan_date, reason FROM execution_log WHERE status='executed'", con)
    except Exception:
        df = pd.DataFrame(columns=["well", "plan_date", "reason"])
    con.close()
    wset = set(map(str, wells))
    if not len(df):
        return {}
    df = df[df["well"].isin(wset)].copy()
    df["plan_date"] = pd.to_datetime(df["plan_date"], errors="coerce")
    df["reason"] = df["reason"].fillna("").astype(str).str.upper()
    return {w: list(zip(grp["plan_date"], grp["reason"])) for w, grp in df.groupby("well")}

def sch_latest(wells):
    """xlookup ke execution_log: ambil schedule_date_test TERAKHIR + status per well.
    Return dict well -> (tanggal_str, label) dengan label COMP/NCMP/PENDING."""
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT well_name AS well, status, plan_date FROM execution_log "
                         "WHERE status IN ('executed','ncmp','pending')", con)
    except Exception:
        df = pd.DataFrame(columns=["well", "status", "plan_date"])
    con.close()
    wset = set(map(str, wells))
    if not len(df):
        return {}
    df = df[df["well"].isin(wset)].copy()
    if not len(df):
        return {}
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
        cr = cols.get("REASON") or cols.get("COMMENT IF NOT COMPLETE")
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
        skip_date += int(w["date"].isna().sum())
        valid = (~w["well"].isin(["", "nan"])) & (w["date"].notna())

        # PENDING: punya schedule_date_test tapi STATUS kosong/tak dikenal → disisihkan
        wp = w[valid & (w["stat"] == "")].copy()
        skip_status += int(((w["stat"] == "") & ~valid).sum())
        if len(wp):
            wp["plan_date"] = wp["date"].dt.date.astype(str)
            prows = list(zip(wp["plan_date"], wp["well"], wp["unit"],
                             ["pending"] * len(wp), [""] * len(wp), [now] * len(wp)))
            # WHERE: jangan timpa hasil COMP/NCMP yang sudah ada utk (well, tanggal) yg sama
            con.executemany("""INSERT INTO execution_log(plan_date,well_name,unit,status,reason,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(plan_date,well_name) DO UPDATE SET
                unit=excluded.unit, status=excluded.status, reason=excluded.reason,
                updated_at=excluded.updated_at
                WHERE execution_log.status NOT IN ('executed','ncmp')""", prows)
            n_pend += len(wp)

        # COMP / NCMP
        w = w[valid & (w["stat"] != "")].copy()
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
    return {"comp": n_comp, "ncmp": n_ncmp, "pending": n_pend, "reasons": reasons, "status_seen": status_seen,
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
    pop_col = next((c for c in df.columns if "POP" in str(c).upper() and "DATE" in str(c).upper()), None)
    df["pop_date"] = to_dt(df[pop_col]) if pop_col else pd.NaT
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
def plan(elig, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, current_day=None, elastic_limit=5.0, blocked_units=None, prebooked=None):
    df = elig.reset_index(drop=True).copy()
    df["scheduled"] = False
    df["plan_unit"] = None
    if "forced_unit" not in df.columns: df["forced_unit"] = None
    if "urgency" not in df.columns: df["urgency"] = 0.0
    # Dua-lapis: sumur prioritas layer-1 (prebooked) ikut ke ruang klaster agar reguler bisa menumpang rute-nya
    df["_pre_unit"] = None
    if prebooked is not None and len(prebooked):
        pb = prebooked.copy()
        pb["_pre_unit"] = pb["plan_unit"]
        pb["forced_unit"] = None
        for _c in df.columns:
            if _c not in pb.columns: pb[_c] = None
        df = pd.concat([df, pb[df.columns]], ignore_index=True)

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

    _blk = set(blocked_units) if blocked_units else set()
    avail_remote = [u for u in list(REMOTE_UNITS)[:n_remote] if u not in _blk]
    avail_nonremote = [u for u in list(NONREMOTE_UNITS)[:n_nonremote] if u not in _blk]
    unit_clusters = {u: [] for u in avail_remote + avail_nonremote}
    unassigned = set(df.index)

    def _grow(u, target_fld=None):
        zone_remote = (u in REMOTE_UNITS)
        while len(unit_clusters[u]) < max_wells and unassigned:
            cand_pool = [i for i in unassigned if (area_arr[i] in REMOTE_AREAS) == zone_remote]
            if not cand_pool: break
            c_dists = dist_mat[np.ix_(unit_clusters[u], cand_pool)]
            min_dists = c_dists.min(axis=0)
            max_dists = c_dists.max(axis=0)
            valid_cands = []
            for i_cand, cand_idx in enumerate(cand_pool):
                d_min = min_dists[i_cand]; d_max = max_dists[i_cand]; urg = urg_arr[cand_idx]
                is_same_fld = (target_fld is not None and field_arr[cand_idx] == target_fld)
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
                        best_idx = cand_idx; break
                else:
                    best_idx = cand_idx; break
            if best_idx is not None:
                unit_clusters[u].append(best_idx); unassigned.remove(best_idx)
            else:
                break

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

    # Dua-lapis: tempatkan sumur prioritas layer-1, lalu tumbuhkan unit-nya dengan reguler
    _pre_units = set()
    if df["_pre_unit"].notna().any():
        for idx in df.index[df["_pre_unit"].notna()]:
            u = df.at[idx, "_pre_unit"]
            if u in unit_clusters and idx in unassigned:
                unit_clusters[u].append(idx); unassigned.discard(idx); _pre_units.add(u)
        for u in _pre_units:
            _grow(u)

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

            _grow(u, target_fld)

            assigned_this_round = True
            break
        if not assigned_this_round: break

    for u, c in unit_clusters.items():
        if c:
            df.loc[c, "scheduled"] = True
            df.loc[c, "plan_unit"] = u

    return df

def plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed,
              use_urg, use_dur, early_days=0, elastic_limit=5.0, unit_blackout=None, prebooked=None):
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
    # NW/AWS yang deadline-nya sudah lewat hari pertama horizon → OVERDUE: boleh dijadwalkan ASAP
    overdue_nw = is_nwaws & elig["max_date"].notna() & (elig["max_date"] < days[0])

    for i, day in enumerate(days, start=1):
        win_reg = (elig["min_date"] - early_td <= day)
        win_nw = (elig["min_date"] <= day) & (elig["max_date"] >= day)
        np_ok = np_in & (next_wt <= day) & ~is_nwaws

        is_late = day > elig["max_date"]
        forbid_late = strict_no_late & is_late

        cond_reg = (~is_nwaws) & (win_reg | np_ok | bypass_reg) & ~forbid_late
        cond_nw = is_nwaws & (win_nw | overdue_nw)
        
        pidx = elig.index[rem & (cond_reg | cond_nw)]
        if len(pidx) == 0: continue

        pool = elig.loc[pidx].copy()
        pool["urgency"] = (pool["max_date"] - day).dt.days.fillna(0)
        
        mid = bypass_reg.loc[pidx]
        nw = is_nwaws.loc[pidx]

        pool.loc[mid, "urgency"] = pool.loc[mid, "urgency"].clip(upper=0)
        pool.loc[nw, "urgency"] = pool.loc[nw, "urgency"].clip(upper=0) - 10000

        # Unit MWT tidak tersedia pada hari ini → buang dari pool tersedia
        day_key = pd.Timestamp(day).strftime("%Y-%m-%d")
        blocked = unit_blackout.get(day_key, set()) if unit_blackout else set()
        if blocked and "forced_unit" in pool.columns:
            # sumur yang dipaksa ke unit terblokir ditunda (tunggu hari unit tersedia)
            pool = pool[~(pool["forced_unit"].notna() & pool["forced_unit"].isin(blocked))]
            if len(pool) == 0: continue

        pb_day = None
        if prebooked is not None and len(prebooked):
            _pbd = prebooked[prebooked["day_idx"] == i]
            if len(_pbd): pb_day = _pbd

        pd_ = plan(pool, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, current_day=day, elastic_limit=elastic_limit, blocked_units=blocked, prebooked=pb_day)

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
    sheet_breakin = st.text_input("Sheet Break-In (NW/AWS/Req sisipan)", "BreakIn",
                                  help="Sumur sisipan tengah jadwal. Kosongkan jika tidak dipakai.")

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
raw["is_breakin"] = False

# Break-In: sumur sisipan (NW/AWS/Req) dengan schema sama spt kandidat utama
raw_break = pd.DataFrame()
if sheet_breakin and sheet_breakin.strip():
    try:
        raw_break = load_candidates(up.getvalue(), sheet_breakin.strip())
    except Exception:
        raw_break = pd.DataFrame()
if len(raw_break):
    raw_break["is_breakin"] = True
    # gabung; jika well sudah ada di kandidat utama, baris break-in yang dipakai
    raw = pd.concat([raw[~raw["well"].isin(set(raw_break["well"]))], raw_break],
                    ignore_index=True)
n_breakin_total = int(raw["is_breakin"].fillna(False).sum())

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

    with st.expander("🗑️ Kelola / Hapus SCH_Database"):
        st.caption("SCH_Database (COMP/NCMP + tanda COMP manual) tersimpan permanen di server sampai dihapus — "
                   "bukan sekadar cache. Menghapus akan mengosongkan seluruh riwayat, lalu app membangun ulang "
                   "hanya dari file yang sedang ada di uploader (kosongkan uploader dulu bila ingin benar-benar bersih).")
        _ok = st.checkbox("Saya paham ini menghapus SEMUA log eksekusi (termasuk tanda COMP manual)", key="_confirm_wipe_sch")
        if st.button("Hapus SCH_Database sekarang", disabled=not _ok, use_container_width=True):
            reset_execution_log()
            for _k in ("_compncmp_sig", "_compncmp_summary"):
                st.session_state.pop(_k, None)
            st.cache_data.clear()
            st.success("SCH_Database dihapus. Data akan dibangun ulang dari file yang sedang di-upload (jika ada).")
            st.rerun()

    st.divider()
    ui.section("⚖️ Data Komparasi Manual")
    manual_file = st.file_uploader("Upload Excel Manual Schedule", type=["xlsx", "xlsm"], help="Untuk perbandingan rute before/after di Tab Komparasi")
    manual_sheet = st.text_input("Nama Sheet Manual", "Well Test Schedule")
    
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

    with st.expander("🚫 Unit MWT Tidak Tersedia (per tanggal)", expanded=False):
        _ps = pd.Timestamp(plan_start_date); _ph = pd.Timestamp(per_hi)
        _hz = max(1, min(60, (_ph - _ps).days + 1))
        _blk_days = [(_ps + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(_hz)]
        st.caption("Centang sel **(unit × tanggal)** saat unit MWT tidak beroperasi / tidak bisa menampung "
                   "sumur pada tanggal tsb. Sumur akan dialihkan ke unit lain atau hari lain.")
        _prev_blk = set(tuple(x) for x in st.session_state.get("unit_blackout", []))
        _grid = pd.DataFrame(False, index=ALL_UNITS, columns=_blk_days)
        for (_u, _dk) in _prev_blk:
            if _u in _grid.index and _dk in _grid.columns:
                _grid.loc[_u, _dk] = True
        _grid_disp = _grid.reset_index().rename(columns={"index": "Unit MWT"})
        _ed_blk = st.data_editor(
            _grid_disp, hide_index=True, use_container_width=True,
            key=f"blk_editor_{len(_blk_days)}_{(_blk_days[0] if _blk_days else '')}",
            column_config={"Unit MWT": st.column_config.TextColumn("Unit MWT", disabled=True),
                           **{dk: st.column_config.CheckboxColumn(dk[5:], default=False, help=dk) for dk in _blk_days}},
            disabled=["Unit MWT"])
        _new_blk = [(r["Unit MWT"], dk) for _, r in _ed_blk.iterrows() for dk in _blk_days if bool(r[dk])]
        st.session_state["unit_blackout"] = _new_blk
        if _new_blk:
            st.caption(f"🚫 **{len(_new_blk)}** slot unit-tanggal diblokir.")
            if st.button("Bersihkan semua blokir", key="clr_blk", use_container_width=True):
                st.session_state["unit_blackout"] = []
                st.rerun()
                                    
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

        two_layer = st.checkbox("Optimasi 2-lapis (prioritas → reguler)", value=False,
            help="Lapis 1: optimasi sumur prioritas (NW/AWS/PRQ/ORQ + carry NCMP) lebih dulu. Lapis 2: sumur reguler mengisi sisa kapasitas unit/hari & menumpang rute prioritas. Algoritma sama; hasil lebih mudah diaudit.")
        
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
    # Sumur TS yang dialihkan ke MWT → durasi pengetesan dipaksa 60 menit
    ts2mwt = raw["is_ts"] & raw["area"].isin(ts_unavail)
    raw.loc[ts2mwt, "dur"] = 60

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

executed_log, ncmp_log, pending_log = status_in_period(per_lo, per_hi)
comp_col = set(raw.loc[raw["sch_status"] == "COMP", "well"])
manual_comp = set(st.session_state.get("manual_comp", []))  # ditandai COMP manual oleh user (review SCH)

# ── AWS dua-fase (Option C): auto-transisi AWS1→AWS2 dari POP_Date + riwayat COMP ──
# AWS1 = POP+1..+3, AWS2 = POP+5..+10. COMP di window AWS1 → fase naik ke AWS2 (window di-override).
# Sumur baru dianggap "selesai" (executed) bila AWS2 sudah COMP.
aws_done = set()        # AWS2 selesai → final (executed)
aws_active = {}         # well → (fase, min_date|None, max_date|None); None = pertahankan window Excel
_aws = raw[raw["tipe"] == "AWS"].drop_duplicates("well")
if len(_aws):
    _recs = comp_records(set(_aws["well"]))
    _has_pop_col = "pop_date" in raw.columns
    for _, _w in _aws.iterrows():
        _wn = _w["well"]
        recs = _recs.get(_wn, [])
        reasons = [r for _, r in recs]
        as1 = any(("AS1" in r) or ("AWS1" in r) for r in reasons)
        as2 = any(("AS2" in r) or ("AWS2" in r) for r in reasons)
        pop = _w["pop_date"] if _has_pop_col else pd.NaT
        has_pop = pd.notna(pop)
        orig_cat = str(_w.get("category", "")).upper()
        excel_aws2 = "AWS2" in orig_cat                 # user sudah melabel AWS2 di Excel
        # window utk fase AWS2: kalau Excel SUDAH AWS2 → pertahankan window Excel (sumber kebenaran user);
        # kalau Excel masih AWS1 → bump ke POP+5..+10 (auto-transition window).
        a2_ovr = (None, None) if (excel_aws2 or not has_pop) else (pop + pd.Timedelta(days=5), pop + pd.Timedelta(days=10))
        ds = [pd.Timestamp(d).normalize() for d, _ in recs if pd.notna(d)]
        a1_win = a2_win = False
        if has_pop:
            popn = pd.Timestamp(pop).normalize()
            a1_win = any(popn + pd.Timedelta(days=1) <= d <= popn + pd.Timedelta(days=3) for d in ds)
            a2_win = any(popn + pd.Timedelta(days=5) <= d <= popn + pd.Timedelta(days=10) for d in ds)
        n_comp = len(ds)
        if as2:
            aws_done.add(_wn)                              # Reason AS2 → selesai
        elif as1:
            aws_active[_wn] = ("AWS2", a2_ovr[0], a2_ovr[1])  # Reason AS1 → naik AWS2 (eligible)
        elif (a1_win and a2_win) or n_comp >= 2:
            aws_done.add(_wn)                              # bukti dua fase → selesai
        elif n_comp >= 1:
            aws_active[_wn] = ("AWS2", a2_ovr[0], a2_ovr[1])  # 1 COMP = AWS1 selesai → AWS2 (eligible)
        # else: belum ada COMP → biarkan apa adanya (fase/window dari Excel)
    for _wn, (_ph, _lo, _hi) in aws_active.items():
        _m = raw["well"] == _wn
        if _lo is not None: raw.loc[_m, "min_date"] = _lo
        if _hi is not None: raw.loc[_m, "max_date"] = _hi
        raw.loc[_m, "category"] = _ph
    for _wn, (_ph, _lo, _hi) in aws_active.items():
        _m = raw["well"] == _wn
        if _lo is not None: raw.loc[_m, "min_date"] = _lo
        if _hi is not None: raw.loc[_m, "max_date"] = _hi
        raw.loc[_m, "category"] = _ph

executed = (executed_log | comp_col | manual_comp | aws_done) - set(aws_active.keys())
# COMP utk DASHBOARD = period-scoped (executed_log sudah difilter periode) + kolom SCH + manual.
# aws_done SENGAJA tidak ikut di sini: itu AWS yang kelar lintas-periode (cukup utk exclude jadwal,
# tapi jangan inflate Status Realisasi periode ini). AWS yg COMP di periode ini tetap kebawa via executed_log.
comp_disp_set = (executed_log | comp_col | manual_comp) - set(aws_active.keys())

# PENDING: jadwal sudah ada tapi STATUS belum diisi → disisihkan, jangan dijadwalkan ulang
pending_col = set(raw.loc[raw["sch_status"].isin(["PENDING", "PEND"]), "well"])
pending_set = (set(pending_log["well"]) | pending_col) - executed
pending_sched = dict(zip(pending_log["well"], pending_log["plan_date"]))

ncmp_log = ncmp_log[~ncmp_log["well"].isin(executed | pending_set)].copy()
ncmp_col = set(raw.loc[raw["sch_status"] == "NCMP", "well"]) - executed - pending_set
ncmp_set = (set(ncmp_log["well"]) | ncmp_col) - executed - pending_set

in_raw = set(raw["well"])

# Override OFF→ON untuk sumur NW/AWS terpilih (status Excel mungkin belum terupdate)
nwaws_off_pool = raw[(raw["tipe"].isin(["NW", "AWS"])) & (raw["status"] == "OFF")][
    ["well", "tipe", "category", "field", "area"]].copy()
force_on = set(st.session_state.get("force_on_nwaws", [])) & set(nwaws_off_pool["well"])
if force_on:
    raw.loc[raw["well"].isin(force_on), "status"] = "ON"

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
# Sumur prioritas (NW/AWS/PRQ/ORQ) yang deadline-nya SUDAH lewat start rentang → OVERDUE.
# Ini bukan urgensi rendah; justru harus dijadwalkan PALING dulu (jangan di-drop).
is_prio_c = is_nwaws_c | raw["req_tag"].isin(["PRQ", "ORQ"])
overdue_prio = is_prio_c & raw["max_date"].notna() & (raw["max_date"] < batch_lo)
comp_wells = raw[raw["well"].isin(comp_disp_set)].copy()
pending_wells = raw[raw["well"].isin(pending_set)].copy()
pending_nodata = sorted(pending_set - set(pending_wells["well"]))
nwaws_dropped = raw[is_nwaws_c & ~in_range & ~overdue_prio & (~raw["well"].isin(executed))].copy()
cand = raw[(in_range | is_ncmp | req_force | overdue_prio) & (~raw["well"].isin(executed | pending_set))].copy()

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

# Sumur REGULER (bukan NW/AWS/PRQ/ORQ/carry-NCMP) yang window min-max-nya di LUAR rentang periode
# → urgensi FLEKSIBEL: tak wajib dites di awal, boleh kapan saja dalam rentang (isi celah).
# (Kebalikan sumur prioritas overdue yang justru harus paling dulu.)
_prio_u = nwaws | mid_prio | elig_all["req_tag"].isin(["PRQ", "ORQ"])
_win_outside = (elig_all["max_date"] < batch_lo) | (elig_all["min_date"] > batch_hi)
_flex = (~_prio_u) & elig_all["max_date"].notna() & _win_outside
_span = max(int((week_hi - week_lo).days), 1)
elig_all.loc[_flex, "urgency"] = _span

elig = elig_all[elig_all["has_coord"]].copy()
nocoord = elig_all[~elig_all["has_coord"]].copy()

# ── Unit MWT Tidak Tersedia per Tanggal (blackout) — input ada di sidebar ───
unit_blackout_by_day = {}
for (_u, _dk) in st.session_state.get("unit_blackout", []):
    unit_blackout_by_day.setdefault(_dk, set()).add(_u)

# ── Rollout Execution Framework ────────────────────────────────────────────
if len(elig):
    if two_layer:
        # Lapis 1: sumur prioritas (NW/AWS/PRQ/ORQ + carry NCMP) dioptimasi lebih dulu
        _prio_mask = elig["is_nwaws"].fillna(False) | elig["req_tag"].isin(["PRQ", "ORQ"]) | elig["carry_ncmp"].fillna(False)
        prio_elig = elig[_prio_mask].copy()
        reg_elig = elig[~_prio_mask].copy()
        if len(prio_elig):
            wk_prio = plan_week(prio_elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, unit_blackout=unit_blackout_by_day)
        else:
            wk_prio = prio_elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
        # Lapis 2: reguler mengisi sisa kapasitas; sumur prioritas terjadwal jadi 'prebooked'
        pb = wk_prio[wk_prio["scheduled"]].copy()
        if len(reg_elig):
            wk_reg = plan_week(reg_elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, unit_blackout=unit_blackout_by_day, prebooked=pb if len(pb) else None)
        else:
            wk_reg = reg_elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
        week_df = pd.concat([wk_prio, wk_reg], ignore_index=True)
    else:
        week_df = plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, unit_blackout=unit_blackout_by_day)
else:
    week_df = elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
if len(nocoord):
    noc = nocoord.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
    week_df = pd.concat([week_df, noc], ignore_index=True)

week_df["zone"] = np.where(week_df["area"].isin(REMOTE_AREAS), "remote", "non-remote")
week_df["manual"] = False

man = st.session_state.get("manual_assign", {})
zone_rejects = []
if man:
    # pool master semua kandidat (utk inject sumur yg belum ada di week_df: luar window/COMP/dll)
    _master = raw.drop_duplicates("well").set_index("well")
    _dtcols = [c for c in week_df.columns if pd.api.types.is_datetime64_any_dtype(week_df[c])]
    _present = set(week_df["well"])
    _inject = []
    for w in man:
        if w not in _present and w in _master.index:
            base = _master.loc[w]
            newrow = {c: (base[c] if c in _master.columns else np.nan) for c in week_df.columns}
            newrow["well"] = w
            newrow["zone"] = "remote" if str(base.get("area")) in REMOTE_AREAS else "non-remote"
            newrow["manual"] = False
            newrow["scheduled"] = False
            if "urgency" in week_df.columns: newrow["urgency"] = 0
            _inject.append(newrow)
    if _inject:
        week_df = pd.concat([week_df, pd.DataFrame(_inject)], ignore_index=True)
        for c in _dtcols:  # concat dgn NaN bisa merusak dtype datetime → paksa balik
            week_df[c] = pd.to_datetime(week_df[c], errors="coerce")
    zone_rejects = []
    for w, info in list(man.items()):
        m = week_df["well"] == w
        if not m.any(): continue
        di = int(info["day_idx"])
        if di < 1 or di > horizon: continue
        # Batasan zona: unit remote (Bangko/Balam) hanya utk sumur remote; non-remote utk Bekasap/Libo dst.
        _u = info["unit"]
        _uzone = "remote" if _u in REMOTE_UNITS else "non-remote"
        _wzone = str(week_df.loc[m, "zone"].iloc[0])
        if _uzone != _wzone:
            zone_rejects.append((w, _wzone, _u, _uzone))
            st.session_state.get("manual_assign", {}).pop(w, None)  # buang assignment lintas-zona (self-heal)
            continue
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

# Rincian durasi tes terjadwal (60 vs 30 menit)
_dur_sched = pd.to_numeric(scheduled_all["dur"], errors="coerce") if len(scheduled_all) else pd.Series(dtype=float)
n_dur60 = int((_dur_sched == 60).sum())
n_dur30 = int((_dur_sched == 30).sum())
n_dur_other = int(total_scheduled - n_dur60 - n_dur30)

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
_dur_caption = (f"🗓️ **{total_scheduled} sumur terjadwal** — "
                f"🕐 tes 60 menit: **{n_dur60}** · 🕧 tes 30 menit: **{n_dur30}**"
                + (f" · durasi lain: **{n_dur_other}**" if n_dur_other else ""))
st.caption(_dur_caption)
st.caption(f"🔌 **WELLS OFF ({len(off_wells)})** = sumur OFF yang jadi kandidat & di-skip **di siklus ini**. "
           f"Total semua sumur OFF di master data: **{len(master_off_wells)}** — lihat daftar lengkapnya di tab "
           f"**⭐ Prioritas & Status Khusus**.")

# ── Main Workspace Tabs ────────────────────────────────────────────────────
def _fv(x):
    """Format nilai field/area: rapikan NaN/kosong jadi '-'."""
    return "-" if (x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "" or str(x).lower() == "nan") else str(x)

def _comp_review_panel(df_src, key, only_hits=False):
    """Tabel review: xlookup SCH (tanggal+status terakhir) vs window min-max, +
    checklist 'Tandai COMP'. Yang dicentang lalu di-proses -> dikeluarkan dari jadwal."""
    if not len(df_src):
        st.caption("Tidak ada sumur untuk direview.")
        return
    look = sch_latest(df_src["well"].tolist())
    rows = []
    for _, w in df_src.iterrows():
        hit = look.get(str(w["well"]))
        rows.append({
            "Tandai COMP": False, "Well": w["well"], "Field": _fv(w.get("field")),
            "Min Date": w["min_date"].strftime("%Y-%m-%d") if pd.notna(w.get("min_date")) else "-",
            "Max Date": w["max_date"].strftime("%Y-%m-%d") if pd.notna(w.get("max_date")) else "-",
            "SCH Test Terakhir": hit[0] if hit else "-",
            "Status SCH": hit[1] if hit else "(tidak ada di SCH)",
        })
    rev = pd.DataFrame(rows)
    if only_hits:
        rev = rev[rev["Status SCH"] != "(tidak ada di SCH)"]
    if not len(rev):
        st.caption("Tidak ada sumur yang punya catatan di SCH_Database untuk direview.")
        return
    rev = rev.sort_values(["Status SCH", "Well"])
    st.caption("Bandingkan **SCH Test Terakhir** & **Status SCH** dengan window **Min–Max**. "
               "Centang sumur yang sudah dianggap **COMP**, lalu klik proses — sumur tsb dikeluarkan dari jadwal & tidak di-replan.")

    # ── Filter: Status SCH + Tanggal (single / rentang) ────────────────────
    fcol1, fcol2, fcol3 = st.columns([1.4, 1, 1.6])
    with fcol1:
        stat_opts = sorted(rev["Status SCH"].unique().tolist())
        stat_pick = st.multiselect("Filter Status SCH", stat_opts, default=stat_opts, key=key + "_fs")
    parsed_all = pd.to_datetime(rev["SCH Test Terakhir"], format="%Y-%m-%d", errors="coerce")
    has_dates = bool(parsed_all.notna().any())
    with fcol2:
        dmode = st.selectbox("Filter Tgl SCH", ["Semua", "Single", "Rentang"],
                             key=key + "_dm", disabled=not has_dates)
    sel_date = None
    with fcol3:
        if has_dates and dmode != "Semua":
            dmin = parsed_all.min().date(); dmax = parsed_all.max().date()
            if dmode == "Single":
                sel_date = st.date_input("Tanggal SCH", value=dmax, min_value=dmin, max_value=dmax, key=key + "_d1")
            else:
                sel_date = st.date_input("Rentang Tgl SCH", value=(dmin, dmax), min_value=dmin, max_value=dmax, key=key + "_d2")

    view = rev[rev["Status SCH"].isin(stat_pick)].copy()
    if has_dates and dmode != "Semua" and sel_date is not None:
        pv = pd.to_datetime(view["SCH Test Terakhir"], format="%Y-%m-%d", errors="coerce").dt.date
        if dmode == "Single":
            lo = hi = sel_date
        elif isinstance(sel_date, (list, tuple)):
            lo, hi = (sel_date[0], sel_date[-1]) if len(sel_date) >= 2 else (sel_date[0], sel_date[0])
        else:
            lo = hi = sel_date
        view = view[pv.notna() & (pv >= lo) & (pv <= hi)]

    st.caption(f"Menampilkan **{len(view)}** dari {len(rev)} sumur sesuai filter.")
    if not len(view):
        st.info("Tidak ada sumur yang cocok dengan filter.")
        return

    sel_all = st.checkbox(f"✔ Centang semua hasil filter ({len(view)} sumur)", key=key + "_all",
                          help="Mencentang semua sumur yang sedang tampil. Masih bisa di-uncheck satu per satu.")
    view = view.copy()
    if sel_all:
        view["Tandai COMP"] = True

    edited = st.data_editor(
        view, hide_index=True, use_container_width=True, key=f"{key}_ed_{int(sel_all)}",
        column_config={"Tandai COMP": st.column_config.CheckboxColumn("✔ COMP?", default=False)},
        disabled=["Well", "Field", "Min Date", "Max Date", "SCH Test Terakhir", "Status SCH"])
    n_sel = int((edited["Tandai COMP"] == True).sum())
    if st.button(f"✅ Proses: Tandai COMP & keluarkan dari jadwal ({n_sel} dipilih)", key=key + "_btn", type="primary"):
        sel = edited[edited["Tandai COMP"] == True]["Well"].tolist()
        if sel:
            st.session_state.setdefault("manual_comp", [])
            for w in sel:
                if w not in st.session_state["manual_comp"]:
                    st.session_state["manual_comp"].append(w)
            st.rerun()
        else:
            st.warning("Belum ada sumur yang dicentang.")

tab_guide, tab_sched, tab_map, tab_matrix, tab_cart, tab_sch, tab_diagnostics, tab_priority, tab_export, tab_compare = st.tabs([
    "📘 Panduan", "📅 Jadwal Operasional", "🗺️ Peta Rute", "📊 Matriks Deviasi", "🛒 Cart Manual",
    "🗃️ SCH Database", "📏 Analisis Jarak", "⭐ Prioritas & Status Khusus", "📤 Export", "⚖️ Komparasi"
])

with tab_guide:
    guide.render_guide()

with tab_sched:
    if len(scheduled_all) == 0:
        st.info("Belum ada jadwal yang berhasil dialokasikan pada siklus ini.")
    else:
        # label dgn field/area + status utk panel "Tambahkan Sumur"
        def _add_label(r):
            tags = []
            if bool(r.get("is_breakin", False)): tags.append("BREAK-IN")
            if not bool(r.get("has_coord", True)): tags.append("no-coord")
            _w = r["well"]
            if _w in executed: tags.append("COMP")
            elif _w in pending_set: tags.append("PENDING")
            else:
                _mn, _mx = r.get("min_date"), r.get("max_date")
                if pd.notna(_mx) and _mx < batch_lo: tags.append("lewat window")
                elif pd.notna(_mn) and _mn > batch_hi: tags.append("blm masuk window")
            t = ("  ·  " + " · ".join(tags)) if tags else ""
            return f"{r['well']}  —  {_fv(r.get('field'))} / {_fv(r.get('area'))}{t}"
        # sumber: SEMUA kandidat yg lolos filter area/MPAS (bukan hanya leftover), minus yg sudah terjadwal
        _sched_now = set(scheduled_all["well"]) if len(scheduled_all) else set()
        _add_src = raw.drop_duplicates("well")
        _add_src = _add_src[~_add_src["well"].isin(_sched_now)]
        _add_map = {_add_label(r): r["well"] for _, r in _add_src.iterrows()} if len(_add_src) else {}

        for day_idx, day_date in enumerate(days, 1):
            day_data = scheduled_all[scheduled_all["day_idx"] == day_idx]
            if len(day_data) == 0: continue
            
            ui.day_header(f"Hari Ke-{day_idx}", day_date.strftime("%A, %d %b"), 
                          units=day_data["plan_unit"].nunique(), wells=len(day_data))
            
            with st.expander(f"⚙️ Atur Manual Sumur Hari Ke-{day_idx}"):
                ca1, ca2 = st.columns(2)
                with ca1:
                    ui.section("➖ Keluarkan Sumur", eyebrow="Batal jadwalkan dari hari ini (per unit)")
                    day_units = sorted(day_data["plan_unit"].dropna().unique().tolist())
                    rm_unit = st.selectbox("Pilih Unit:", ["(Semua Unit)"] + day_units, key=f"rm_u_{day_idx}")
                    if rm_unit == "(Semua Unit)":
                        _rm_map = {f"{r['well']}  —  {r['plan_unit']}": r["well"] for _, r in day_data.iterrows()}
                    else:
                        _rm_map = {r["well"]: r["well"] for _, r in day_data[day_data["plan_unit"] == rm_unit].iterrows()}
                    st.caption(f"{len(_rm_map)} sumur terjadwal di {rm_unit}.")
                    to_rm_lbl = st.multiselect("Pilih sumur:", sorted(_rm_map.keys()), key=f"rm_w_{day_idx}_{rm_unit}")
                    to_rm = [_rm_map[l] for l in to_rm_lbl]
                    if st.button("Keluarkan", key=f"btn_rm_{day_idx}", use_container_width=True):
                        st.session_state.setdefault("manual_unassign", [])
                        for w in to_rm:
                            if w in st.session_state.get("manual_assign", {}):
                                del st.session_state["manual_assign"][w]
                            if w not in st.session_state["manual_unassign"]:
                                st.session_state["manual_unassign"].append(w)
                        st.rerun()
                with ca2:
                    ui.section("➕ Tambahkan Sumur", eyebrow="Cari & tinjau field sebelum masukkan")
                    to_add_lbl = st.multiselect("Cari sumur (nama — field / area):",
                                                sorted(_add_map.keys()), key=f"add_w_{day_idx}")
                    to_add = [_add_map[l] for l in to_add_lbl]
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
    # ── Dashboard: Sumur Tanpa Koordinat ───────────────────────────────────
    if len(nocoord):
        _nb = int(nocoord["is_breakin"].fillna(False).sum()) if "is_breakin" in nocoord.columns else 0
        _title = f"📍 Sumur Tanpa Koordinat: {len(nocoord)} sumur" + (f" · {_nb} break-in" if _nb else "")
        with st.expander(_title, expanded=True):
            st.caption("Sumur ini tidak punya koordinat sehingga **tidak bisa di-route otomatis**. "
                       "Cari namanya di kolom pencarian peta untuk verifikasi, atau assign manual di tab "
                       "**Cart Manual** (panel *Break-In & Tanpa Koordinat*) / kartu unit harian — "
                       "dengan meninjau field-nya.")
            _nc_cols = ["well", "field", "area", "subarea", "category", "tipe", "max_date"]
            nc_show = nocoord[[c for c in _nc_cols if c in nocoord.columns]].copy()
            if "is_breakin" in nocoord.columns:
                nc_show.insert(1, "break_in", np.where(nocoord["is_breakin"].fillna(False).values, "✅", ""))
            if "max_date" in nc_show.columns:
                nc_show["max_date"] = nc_show["max_date"].dt.strftime("%Y-%m-%d")
            nc_show = nc_show.rename(columns={
                "well": "Well", "break_in": "Break-In", "field": "Field", "area": "Area",
                "subarea": "Sub-area", "category": "Kategori", "tipe": "Tipe", "max_date": "Deadline"})
            _sort_keys = [c for c in ["Field", "Well"] if c in nc_show.columns]
            st.dataframe(nc_show.sort_values(_sort_keys) if _sort_keys else nc_show,
                         use_container_width=True, hide_index=True)

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

    n_miss_coord = int(missed["has_coord"].fillna(False).sum()) if len(missed) else 0
    show_miss = st.checkbox(f"📌 Tampilkan Miss Deadline di peta ({n_miss_coord} sumur berkoordinat)",
                            value=False,
                            help="Zoom & tandai sumur miss deadline: nama, tipe (NW/AWS/PRQ/ORQ/RTN), dan window min–max.")
    miss_map = missed[missed["has_coord"].fillna(False)].copy() if len(missed) else leftover.iloc[0:0]

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
            foc = search_hits
            zoom_lvl = 13.5 if len(search_hits) == 1 else 11.0

            # bullet (lingkaran) sumur hasil pencarian: kuning + ring tebal
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=search_hits.copy(),
                get_position=["lon", "lat"],
                get_fill_color=[255, 215, 0],
                get_radius=170,
                get_line_color=[40, 40, 40],
                get_line_width=5,
                line_width_min_pixels=2,
                stroked=True, filled=True, pickable=True, opacity=0.95
            ))
            layers.append(pdk.Layer(
                "TextLayer",
                data=search_hits.copy(),
                get_position=["lon", "lat"],
                get_text="well",
                get_size=75,
                get_color=[0, 0, 0],  # <--- UBAH JADI HITAM [0, 0, 0] DI SINI
                get_pixel_offset=[0, -45],
                font_family="Inter",
                font_weight="bold",
                pickable=False
            ))

    if len(pmap) or len(prev) or len(search_hits) or field_block or (show_miss and len(miss_map)):
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

        # ── Overlay Miss Deadline: pin warna kategori + nama + tipe & window ──
        if show_miss and len(miss_map):
            mm = _tipcols(miss_map.copy())
            _rt = mm["req_tag"].fillna("") if "req_tag" in mm.columns else pd.Series("", index=mm.index)
            def _catv(tp, rt):
                if tp == "NW": return "NW"
                if tp == "AWS": return "AWS"
                if rt == "PRQ": return "PRQ"
                if rt == "ORQ": return "ORQ"
                return "RTN"
            mm["cat"] = [_catv(tp, rt) for tp, rt in zip(mm["tipe"], _rt)]
            CAT_COL = {"NW": [107, 79, 216], "AWS": [230, 178, 58], "RTN": [31, 157, 114], "PRQ": [59, 130, 246], "ORQ": [214, 71, 58]}
            mm["mcol"] = mm["cat"].map(lambda c: CAT_COL.get(c, [120, 120, 120]))
            mm["tipe"] = mm["cat"]                       # tooltip {tipe} -> kategori
            mm["ket"] = "MISS DEADLINE"
            mm["plan_unit"] = mm["unit"].fillna("—") if "unit" in mm.columns else "—"
            mm["sub"] = "[" + mm["cat"].astype(str) + "] " + mm["min_str"].astype(str) + " → " + mm["max_str"].astype(str)
            layers.append(pdk.Layer("ScatterplotLayer", data=mm, get_position=["lon", "lat"],
                get_fill_color="mcol", get_radius=160, get_line_color=[210, 30, 30], get_line_width=4,
                line_width_min_pixels=2, stroked=True, filled=True, pickable=True, opacity=0.95))
            layers.append(pdk.Layer("TextLayer", data=mm, get_position=["lon", "lat"], get_text="well",
                get_size=58, get_color=[190, 20, 20], get_pixel_offset=[0, -42],
                font_family="Inter", font_weight="bold", pickable=False))
            layers.append(pdk.Layer("TextLayer", data=mm, get_position=["lon", "lat"], get_text="sub",
                get_size=32, get_color=[40, 40, 40], get_pixel_offset=[0, 30],
                font_family="Inter", font_weight="bold", pickable=False))

        if len(search_hits):
            foc = search_hits
            zoom_lvl = 13.5 if len(search_hits) == 1 else 11.0

            # bullet (lingkaran) sumur hasil pencarian: kuning + ring tebal
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=search_hits.copy(),
                get_position=["lon", "lat"],
                get_fill_color=[255, 215, 0],
                get_radius=170,
                get_line_color=[40, 40, 40],
                get_line_width=5,
                line_width_min_pixels=2,
                stroked=True, filled=True, pickable=True, opacity=0.95
            ))
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
            
        elif show_miss and len(miss_map):
            foc = miss_map
            zoom_lvl = 13.0 if len(miss_map) == 1 else 11.5
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
        miss_note = "  📌 Miss Deadline = pin ring merah + label nama, tipe, & window." if (show_miss and len(miss_map)) else ""
        st.caption(f"💡 {legend}. Ring Merah=NW, Oranye=AWS. Garis biru menghubungkan sequence rute TSP antar sumur.{miss_note}")

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

        with st.expander("🔎 Review Miss Deadline vs SCH_Database — tandai COMP manual", expanded=False):
            _comp_review_panel(missed, key="rev_miss")

    with st.expander(f"🔍 Evaluasi Pengecualian Kandidat (Ter-Skip) - Klik Untuk Expand"):
        elig_set = set(elig_all["well"])
        out = raw[~raw["well"].isin(elig_set)].copy()
        no_date = out[out["min_date"].isna() | out["max_date"].isna()]
        is_comp = out[out["well"].isin(executed)]
        is_pend = out[out["well"].isin(pending_set)]
        is_off = out[out["status"] == "OFF"]
        is_woff = out[out["well"].isin(woff_set)]
        nw_out = out[out["is_nwaws"].fillna(False) & ~((out["min_date"] <= batch_hi) & (out["max_date"] >= batch_lo))]
        accounted = (set(no_date["well"]) | set(is_comp["well"]) | set(is_pend["well"]) | set(is_off["well"]) | set(is_woff["well"]) | set(nw_out["well"]))
        win_out = out[~out["well"].isin(accounted) & out["min_date"].notna() & out["max_date"].notna()]
        st.markdown(
            f"- **Formula Excel Kosong (Min/Max Date)**: {len(no_date)} sumur dibuang karena window tak terbaca.\n"
            f"- **Diluar Rentang Siklus**: {len(win_out)} sumur due di luar horizon. Lebarkan periode jika ingin disertakan.\n"
            f"- **NW/AWS Diluar Siklus**: {len(nw_out)} sumur.\n"
            f"- **PENDING (jadwal ada, status kosong)**: {len(is_pend)} sumur disisihkan menunggu hasil.\n"
            f"- **Status Exclude**: {len(is_comp)} COMP, {len(is_off)} OFF, {len(is_woff)} NCMP-WOFF.")

with tab_cart:
    # ── 🚑 Rescue Miss-Deadline (Tahap 2) ─────────────────────────────────
    ui.section("🚑 Rescue Miss-Deadline (Tahap 2)", eyebrow="Gabungkan sumur miss ke rute existing TERDEKAT (distance-first, overflow ≤8)")
    SOFT_CAP = 8
    _miss_c = int(missed["has_coord"].fillna(True).sum()) if len(missed) else 0
    st.caption(f"**{len(missed)}** sumur miss deadline ({_miss_c} berkoordinat). Tiap sumur **digabung ke rute unit+hari "
               f"yang sudah ada & paling dekat** dalam window-nya (utamakan jarak), kapasitas dilonggarkan sampai "
               f"**{SOFT_CAP}**/unit/hari. Sumur yang jaraknya melebihi batas detour **dibiarkan miss** biar total jarak tidak meledak.")
    _detour_cap = st.slider("Batas jarak ke rute terdekat (km) — sumur lebih jauh dibiarkan miss", 2, 100, 20, key="rescue_detour")
    rc1, rc2 = st.columns([1, 1])
    _do_rescue = rc1.button("🚑 Jalankan Rescue Miss-Deadline", type="primary", disabled=(len(missed) == 0), key="btn_rescue")
    _do_cancel = rc2.button("↩️ Batal Rescue", disabled=(not st.session_state.get("rescued_wells")), key="btn_rescue_cancel")

    if _do_cancel:
        for _w in st.session_state.get("rescued_wells", []):
            st.session_state.get("manual_assign", {}).pop(_w, None)
        st.session_state["rescued_wells"] = []
        st.rerun()

    if _do_rescue and len(missed):
        # titik rute & jumlah per (unit, hari) dari jadwal saat ini
        _route_pts, _counts = {}, {}
        for _, _r in scheduled_all.iterrows():
            _k = (_r["plan_unit"], int(_r["day_idx"]))
            _counts[_k] = _counts.get(_k, 0) + 1
            if bool(_r.get("has_coord", True)) and pd.notna(_r.get("lat")):
                _route_pts.setdefault(_k, []).append((float(_r["lat"]), float(_r["lon"])))
        _assign = dict(st.session_state.get("manual_assign", {}))
        _rescued, _added_km, _skip_far = [], 0.0, 0
        for _, _w in missed.sort_values(["urgency", "max_date"]).iterrows():
            _wn = _w["well"]
            _pool = REMOTE_UNITS if str(_w.get("area", "")).upper() in REMOTE_AREAS else NONREMOTE_UNITS
            _cand_days = [di for di in range(1, horizon + 1)
                          if (pd.isna(_w["min_date"]) or days[di - 1] >= _w["min_date"])
                          and (pd.isna(_w["max_date"]) or days[di - 1] <= _w["max_date"])]
            if not _cand_days:
                continue
            _has_c = bool(_w.get("has_coord", True)) and pd.notna(_w.get("lat"))
            _best = None  # (score, unit, day, dist)
            for _di in _cand_days:
                for _u in _pool:
                    _k = (_u, _di)
                    _pts = _route_pts.get(_k, [])
                    if not _pts:           # hanya gabung ke rute yang SUDAH ada
                        continue
                    _cnt = _counts.get(_k, 0)
                    if _cnt >= SOFT_CAP:    # overflow lunak maksimal 8
                        continue
                    _dist = float(np.min(haversine_km(_w["lat"], _w["lon"],
                                  np.array([p[0] for p in _pts]), np.array([p[1] for p in _pts])))) if _has_c else 0.0
                    _score = _dist + _cnt * 0.001   # distance-first; isi unit cuma tiebreaker halus
                    if _best is None or _score < _best[0]:
                        _best = (_score, _u, _di, _dist)
            if _best is None:
                continue
            _, _bu, _bd, _bdist = _best
            if _has_c and _bdist > _detour_cap:   # terlalu jauh → biarkan miss
                _skip_far += 1
                continue
            _assign[_wn] = {"unit": _bu, "day_idx": _bd}
            _counts[(_bu, _bd)] = _counts.get((_bu, _bd), 0) + 1
            if _has_c:
                _route_pts.setdefault((_bu, _bd), []).append((float(_w["lat"]), float(_w["lon"])))
                _added_km += 2.0 * _bdist          # estimasi out-and-back
            _rescued.append(_wn)
        st.session_state["manual_assign"] = _assign
        st.session_state["rescued_wells"] = _rescued
        st.session_state["rescue_added_km"] = round(_added_km, 1)
        st.session_state["rescue_skip_far"] = _skip_far
        st.rerun()

    _resc = st.session_state.get("rescued_wells", [])
    if _resc:
        _placed = [w for w in _resc if w in set(scheduled_all["well"])]
        _akm = st.session_state.get("rescue_added_km", 0.0)
        _sf = st.session_state.get("rescue_skip_far", 0)
        st.success(f"✅ {len(_placed)} sumur miss tersisipkan · estimasi tambahan jarak **~{_akm} km**"
                   + (f" · {_sf} sumur dilewati (terlalu jauh)" if _sf else ""))
        # unit+hari yang melebihi kapasitas normal → kandidat take-out (urgensi terendah, bukan yg baru di-rescue)
        _over = []
        for (_u, _d), _g in scheduled_all.groupby(["plan_unit", "day_idx"]):
            if len(_g) > max_wells:
                _cand = _g[~_g["well"].isin(_resc)].sort_values("urgency", ascending=False)
                for _, _rr in _cand.head(len(_g) - max_wells).iterrows():
                    _over.append({"Unit": _u, "Hari": int(_d), "Well": _rr["well"],
                                  "Kategori": _fv(_rr.get("category")), "Urgensi": int(_rr.get("urgency", 0)),
                                  "Isi Unit": f"{len(_g)}/{max_wells}"})
        if _over:
            st.warning(f"Beberapa unit lewat kapasitas normal ({max_wells}/hari). Kandidat di-take-out (urgensi terendah, "
                       "sumur prioritas NW/AWS otomatis dikecualikan):")
            st.dataframe(pd.DataFrame(_over).sort_values(["Unit", "Hari", "Urgensi"], ascending=[True, True, False]),
                         use_container_width=True, hide_index=True)
            st.caption("Take-out lewat panel **Keluarkan Sumur** di tab Jadwal Operasional (pilih unit), atau biarkan jika overflow oke.")
    st.divider()

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
    if zone_rejects:
        _zr = ", ".join(f"**{w}** ({wz}) → {u}" for w, wz, u, uz in zone_rejects)
        st.warning(f"⛔ {len(zone_rejects)} assignment lintas-zona ditolak & dibatalkan otomatis. "
                   f"Unit remote (Bangko/Balam) hanya untuk sumur remote; unit non-remote untuk Bekasap/Libo dst. "
                   f"Termasuk sumur gas. Ditolak: {_zr}")
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

            _tp = str(w.get("tipe", "")).upper()
            _catg = str(w.get("category", "")).lower()
            if _tp == "NW": _grp = "NW"
            elif _tp == "AWS": _grp = "AWS"
            elif "manual" in _catg: _grp = "Add Manual"
            else: _grp = "Regular"

            recs.append({
                "Pilih": False, "Well": w['well'], "Tipe": _grp, "Kategori": _fv(w.get('category')),
                "Deadline": w['max_date'].strftime('%Y-%m-%d') if pd.notna(w['max_date']) else '-',
                "Target Unit": target_u, "Hari ke-": target_d, "Isi Keranjang": basket_str, "Jarak Kedekatan (km)": round(best_dist, 1),
                "Status": "⚠️ Miss Deadline" if w['well'] in missed['well'].values else "Sisa Pool"
            })

    if recs:
        rec_df = pd.DataFrame(recs)
        f0, f1, f2, f3 = st.columns([1.6, 1.8, 1.8, 1.4])
        flt_cat = f0.selectbox("Filter Tipe:", ["Semua", "NW", "AWS", "Regular", "Add Manual"], key="cart_cat")
        flt_unit = f1.selectbox("Filter Unit Armada:", ["Semua Unit"] + ALL_UNITS, key="cart_u")
        flt_day = f2.selectbox("Filter Hari Kerja Horizon:", ["Semua Hari"] + list(range(1, horizon + 1)), key="cart_d")
        flt_miss = f3.checkbox("Hanya Miss Deadline", value=True, key="cart_m")

        view_df = rec_df.copy()
        if flt_cat != "Semua": view_df = view_df[view_df["Tipe"] == flt_cat]
        if flt_unit != "Semua Unit": view_df = view_df[view_df["Target Unit"] == flt_unit]
        if flt_day != "Semua Hari": view_df = view_df[view_df["Hari ke-"] == int(flt_day)]
        if flt_miss: view_df = view_df[view_df["Status"].str.contains("Miss Deadline")]

        view_df = view_df.sort_values(["Hari ke-", "Target Unit", "Jarak Kedekatan (km)"])

        sel_all = st.checkbox(f"✅ Pilih semua item pada filter ini ({len(view_df)} sumur)", key="cart_all")
        if sel_all and len(view_df):
            view_df = view_df.assign(Pilih=True)

        edited_rec = st.data_editor(
            view_df, hide_index=True, use_container_width=True,
            column_config={
                "Pilih": st.column_config.CheckboxColumn("Masukin Armada?", default=False),
                "Tipe": st.column_config.TextColumn("Tipe"),
                "Target Unit": st.column_config.SelectboxColumn("Ubah Unit Logistik", options=ALL_UNITS),
                "Hari ke-": st.column_config.NumberColumn("Ubah Hari Horizon", min_value=1, max_value=horizon)
            },
            disabled=["Well", "Tipe", "Deadline", "Isi Keranjang", "Jarak Kedekatan (km)", "Status", "Kategori"]
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

    # ── Break-In & Sumur Tanpa Koordinat — assign manual dgn tinjau field ───
    ui.section("🧩 Break-In & Sumur Tanpa Koordinat", eyebrow="Assign manual dengan meninjau field")
    if len(leftover):
        _bi = leftover["is_breakin"].fillna(False) if "is_breakin" in leftover.columns else pd.Series(False, index=leftover.index)
        _nc = ~leftover["has_coord"].fillna(False) if "has_coord" in leftover.columns else pd.Series(False, index=leftover.index)
        attn = leftover[_bi | _nc].copy()
    else:
        attn = leftover.iloc[0:0]

    if len(attn):
        rows_a = []
        for _, w in attn.iterrows():
            zone = "remote" if str(w.get("area", "")).upper() in REMOTE_AREAS else "non-remote"
            suggest_u = REMOTE_UNITS[0] if zone == "remote" else NONREMOTE_UNITS[0]
            tp = w.get("tipe", "")
            kat = tp if tp in ("NW", "AWS") else (w.get("req_tag", "") or "RTN")
            rows_a.append({
                "Pilih": False, "Well": w["well"],
                "Field": _fv(w.get("field")), "Area": _fv(w.get("area")),
                "Kategori": kat,
                "Break-In": "✅" if bool(w.get("is_breakin", False)) else "",
                "Koordinat": "ada" if bool(w.get("has_coord", True)) else "❌ kosong",
                "Deadline": w["max_date"].strftime("%Y-%m-%d") if pd.notna(w["max_date"]) else "-",
                "Target Unit": suggest_u, "Hari ke-": 1,
            })
        attn_df = pd.DataFrame(rows_a).sort_values(["Break-In", "Field", "Well"], ascending=[False, True, True])
        st.caption("Tinjau **Field/Area** tiap sumur, set Target Unit & Hari, lalu assign. "
                   "Sumur tanpa koordinat tetap bisa dimasukkan (tidak menambah jarak rute).")
        edited_attn = st.data_editor(
            attn_df, hide_index=True, use_container_width=True,
            column_config={
                "Pilih": st.column_config.CheckboxColumn("Assign?", default=False),
                "Target Unit": st.column_config.SelectboxColumn("Unit", options=ALL_UNITS),
                "Hari ke-": st.column_config.NumberColumn("Hari", min_value=1, max_value=horizon),
            },
            disabled=["Well", "Field", "Area", "Kategori", "Break-In", "Koordinat", "Deadline"],
            key="attn_editor")
        if st.button("➕ Assign Break-In / Tanpa Koordinat Terpilih", type="primary", key="attn_btn"):
            sel = edited_attn[edited_attn["Pilih"] == True]
            if not sel.empty:
                st.session_state.setdefault("manual_assign", {})
                st.session_state.setdefault("manual_unassign", [])
                for _, r in sel.iterrows():
                    st.session_state["manual_assign"][r["Well"]] = {"unit": r["Target Unit"], "day_idx": int(r["Hari ke-"])}
                    if r["Well"] in st.session_state["manual_unassign"]:
                        st.session_state["manual_unassign"].remove(r["Well"])
                st.rerun()
    else:
        st.caption("Tidak ada sumur break-in atau tanpa koordinat pada siklus ini.")

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
    tot_pend = len(pending_wells) + len(pending_nodata)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>Total COMP</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:{ui.TEAL_GREEN};'>{tot_comp}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>Total NCMP</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:#E67E22;'>{tot_ncmp}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>NCMP (Dijadwal Ulang)</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:{ui.TEAL};'>{tot_replan}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>NCMP (Sumur OFF / Skip)</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:{ui.RED};'>{tot_woff}</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>PENDING (Belum ada status)</div><div class='wg-disp' style='font-size:24px;font-weight:700;color:#6B4FD8;'>{tot_pend}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Rekonsiliasi: Total NCMP = replan + OFF/skip + (NCMP tanpa baris kandidat)
    tot_nodata = len(ncmp_no_data)
    _bal = tot_ncmp - tot_replan - tot_woff - tot_nodata
    st.caption(
        f"**Rekonsiliasi NCMP:** Total {tot_ncmp} = {tot_replan} dijadwal ulang + {tot_woff} OFF/skip + "
        f"**{tot_nodata} tidak ada baris kandidat** (ke-exclude area mis. LIBO, filter MPAS-only, "
        f"atau memang tak ada di sheet Kandidat)" + (f" + {_bal} lainnya" if _bal else "") + ".")
    if tot_nodata:
        with st.expander(f"🔻 {tot_nodata} NCMP tanpa baris kandidat (tidak bisa di-replan)"):
            st.dataframe(pd.DataFrame({"well": ncmp_no_data}), use_container_width=True, hide_index=True)

    t1, t2, t3, t4 = st.tabs(["✅ Data COMP", "🔁 NCMP (Dijadwalkan Ulang)", "⏸️ NCMP (Skip / OFF)", "⏳ PENDING (Disisihkan)"])
    with t1:
        if len(comp_wells):
            cw = comp_wells[["well", "unit", "subarea", "category", "dur", "sch_status"]].copy()
            cw.insert(1, "sumber", np.where(cw["well"].isin(manual_comp), "✔ Manual",
                                    np.where(cw["well"].isin(comp_col), "SCH-kolom", "SCH-file")))
            st.dataframe(cw.rename(columns={"unit": "unit_asli", "dur": "durasi", "sch_status": "SCH"}), use_container_width=True, hide_index=True)
            if manual_comp:
                if st.button(f"↩️ Batalkan semua tanda COMP manual ({len(manual_comp)})", key="clr_manual_comp"):
                    st.session_state["manual_comp"] = []
                    st.rerun()
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
    with t4:
        st.caption("Sumur ini **sudah punya schedule_date_test tapi STATUS-nya masih kosong** (belum COMP/NCMP). "
                   "Otomatis disisihkan — tidak dijadwalkan ulang sampai hasilnya diisi.")
        if len(pending_wells):
            pend_show = pending_wells[["well", "field", "area", "subarea", "category", "max_date"]].copy()
            pend_show.insert(1, "tgl_jadwal", pend_show["well"].map(pending_sched).fillna("-"))
            pend_show["max_date"] = pend_show["max_date"].dt.strftime("%Y-%m-%d")
            pend_show = pend_show.rename(columns={
                "well": "Well", "tgl_jadwal": "Tgl Jadwal Test", "field": "Field", "area": "Area",
                "subarea": "Sub-area", "category": "Kategori", "max_date": "Deadline"})
            st.dataframe(pend_show.sort_values(["Tgl Jadwal Test", "Well"]), use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada sumur PENDING (semua jadwal sudah ada status COMP/NCMP).")

        # Pending yang TERDETEKSI di SCH tapi tidak ada di pool kandidat
        # (area di-exclude, ter-filter MPAS, atau tidak terdaftar di sheet Kandidat)
        if pending_nodata:
            nd = pd.DataFrame({"Well": pending_nodata})
            nd["Tgl Jadwal Test"] = nd["Well"].map(pending_sched).fillna("-")
            with st.expander(f"⚠️ {len(pending_nodata)} sumur PENDING di SCH tapi tidak ada di pool kandidat"):
                st.caption("Punya schedule_date_test + STATUS kosong di SCH_Database, tapi tidak masuk pool kandidat "
                           "(area di-exclude mis. LIBO, ter-filter MPAS-only, atau tidak terdaftar di sheet Kandidat). "
                           "Tetap disisihkan dari penjadwalan.")
                st.dataframe(nd.sort_values(["Tgl Jadwal Test", "Well"]), use_container_width=True, hide_index=True)

    st.divider()
    ui.section("🔎 Review Eligible vs SCH_Database", eyebrow="xlookup status & tanggal terakhir — tandai COMP manual")
    st.caption("Sumur **eligible** yang punya catatan di SCH_Database (sudah pernah dijadwalkan/dites). "
               "Tinjau window Min–Max vs status terakhirnya, lalu centang yang sudah dianggap COMP — "
               "sumur tsb dikeluarkan dari eligible & tidak dijadwalkan ulang.")
    _comp_review_panel(elig_all, key="rev_elig", only_hits=True)

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

        # ── Dashboard KM / WELL (efisiensi jarak per sumur) ────────────────
        ui.section("Efisiensi Jarak per Sumur (KM/WELL)", eyebrow="Rasio jarak tempuh terhadap jumlah sumur")
        tot_km = float(kdf_an["km"].sum())
        tot_wells = int(kdf_an["Sumur"].sum())
        kmw = (tot_km / tot_wells) if tot_wells else 0.0
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='wg-card' style='padding:14px;text-align:center;'><div class='wg-eyb'>Total Jarak</div><div style='font-size:24px;font-weight:700;color:{ui.TEAL};'>{tot_km:.1f} <span style='font-size:13px;'>km</span></div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='wg-card' style='padding:14px;text-align:center;'><div class='wg-eyb'>Total Sumur Terjadwal</div><div style='font-size:24px;font-weight:700;color:{ui.HEADER_BG};'>{tot_wells}</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='wg-card' style='padding:14px;text-align:center;'><div class='wg-eyb'>Rata-rata KM / WELL</div><div style='font-size:24px;font-weight:700;color:{ui.AMBER};'>{kmw:.2f} <span style='font-size:13px;'>km/well</span></div></div>", unsafe_allow_html=True)

        per_unit = kdf_an.groupby("Unit").agg(**{"Jarak (km)": ("km", "sum"), "Sumur": ("Sumur", "sum")}).reset_index()
        per_unit["KM / WELL"] = (per_unit["Jarak (km)"] / per_unit["Sumur"].clip(lower=1)).round(2)
        per_unit["Jarak (km)"] = per_unit["Jarak (km)"].round(1)
        per_unit = per_unit.sort_values("KM / WELL", ascending=False)
        cda, cdb = st.columns([1.3, 1])
        with cda:
            st.dataframe(per_unit, use_container_width=True, hide_index=True)
        with cdb:
            st.bar_chart(per_unit.set_index("Unit")["KM / WELL"])
        st.caption("KM/WELL tinggi = unit menempuh jarak besar untuk sedikit sumur (rute kurang efisien / sumur tersebar). "
                   "Pakai untuk spot unit yang rutenya boros.")

        ui.section("Tren Jarak Geografis Harian", eyebrow="Total KM & KM/WELL per hari")
        per_day = kdf_an.groupby(["Tanggal"]).agg(km=("km", "sum"), Unit=("Unit", "nunique"), Sumur=("Sumur", "sum")).reset_index()
        per_day["km/sumur"] = (per_day["km"] / per_day["Sumur"].clip(lower=1)).round(2)
        per_day["Tanggal"] = per_day["Tanggal"].astype(str)
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.caption("Total Jarak (km) / hari")
            st.bar_chart(per_day.set_index("Tanggal")["km"])
        with tcol2:
            st.caption("KM / WELL per hari")
            st.bar_chart(per_day.set_index("Tanggal")["km/sumur"])
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

with tab_export:
    ui.section("Export Excel", eyebrow="Unduh jadwal & rute untuk operator lapangan")
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
    else:
        ex2.caption("Pilih **1 tanggal tunggal** di tab Peta Rute untuk mengaktifkan unduhan rute harian.")

with tab_compare:
    ui.section("Komparasi Rute: Manual vs WELLGO", eyebrow="Evaluasi Efisiensi Jarak & Distribusi Harian")
    
    if manual_file is None:
        st.info("💡 Upload file Excel 'Well Test Schedule' manual (.xlsm/.xlsx) di sidebar untuk melihat perbandingan head-to-head.")
    else:
        try:
            # Tambahan: Filter Tanggal khusus tab komparasi
            day_labels_comp = [days[i].strftime("%Y-%m-%d") for i in range(horizon)]
            lbl2idx_comp = {lbl: i + 1 for i, lbl in enumerate(day_labels_comp)}
            
            c_flt_comp, _ = st.columns([3, 1])
            with c_flt_comp:
                comp_sel_labels = st.multiselect("🗓️ Fokus Tanggal Rute (Pilih untuk view komparasi)", day_labels_comp, default=day_labels_comp, key="comp_sel_dates")
            
            if not comp_sel_labels:
                comp_sel_labels = day_labels_comp
            
            comp_sel_idx = sorted(lbl2idx_comp[l] for l in comp_sel_labels)
            
            # 1. Parsing Data Manual
            man_df = pd.read_excel(BytesIO(manual_file.getvalue()), sheet_name=manual_sheet, engine="openpyxl")
            man_df.columns = [str(c).strip().upper() for c in man_df.columns]
            
            if "SCHEDULE DATE" not in man_df.columns or "WELL" not in man_df.columns or "UNIT" not in man_df.columns:
                st.error("Format tabel tidak dikenali. Pastikan file memiliki kolom: WELL, UNIT, SCHEDULE DATE.")
            else:
                # FILTER SINKRONISASI TANGGAL: Gunakan tanggal yang dipilih
                selected_dates = pd.to_datetime(comp_sel_labels).date
                man_df["SCHEDULE DATE"] = pd.to_datetime(man_df["SCHEDULE DATE"], errors="coerce")
                man_df = man_df[man_df["SCHEDULE DATE"].notna()]
                man_df = man_df[man_df["SCHEDULE DATE"].dt.date.isin(selected_dates)]
                
                # Normalisasi Unit & Buang Fasilitas TS
                man_df["UNIT"] = man_df["UNIT"].map(norm_unit)
                man_df = man_df[man_df["UNIT"].astype(str).str.startswith("MPAS")]
                
                if man_df.empty:
                    st.warning(f"Jadwal manual pada tanggal {', '.join(comp_sel_labels)} kosong atau hanya berisi unit non-MPAS (TS).")
                else:
                    # FIX BUG REINDEXING: Tambahkan drop_duplicates("well")
                    spasial_map = field_wells_coord.drop_duplicates(subset=["well"]).set_index("well") if not field_wells_coord.empty else pd.DataFrame()
                    
                    man_df["LAT"] = np.nan
                    man_df["LON"] = np.nan
                    if not spasial_map.empty:
                        valid_wells = man_df["WELL"].isin(spasial_map.index)
                        man_df.loc[valid_wells, "LAT"] = man_df.loc[valid_wells, "WELL"].map(spasial_map["lat"])
                        man_df.loc[valid_wells, "LON"] = man_df.loc[valid_wells, "WELL"].map(spasial_map["lon"])
                    
                    man_valid = man_df[man_df["LAT"].notna() & man_df["LON"].notna()].copy()
                    
                    # 2. Kalkulasi Jarak Manual
                    manual_km = 0.0
                    for (date, unit), group in man_valid.groupby(["SCHEDULE DATE", "UNIT"]):
                        manual_km += route_distance(group["LAT"].values, group["LON"].values)
                    
                    # Data WELLGO khusus untuk komparasi berdasarkan filter
                    comp_disp = scheduled_all[scheduled_all["day_idx"].isin(comp_sel_idx)].copy() if len(scheduled_all) else scheduled_all.copy()
                    
                    # Kalkulasi Jarak WELLGO DINAMIS
                    wellgo_km = 0.0
                    wellgo_wells = len(comp_disp)
                    for (di, dday, unit), sub in comp_disp.groupby(["day_idx", "plan_day", "plan_unit"]):
                        c = sub[sub["has_coord"]]
                        dist_val = route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0
                        wellgo_km += dist_val
                    
                    man_wells = len(man_valid)
                    man_km_well = manual_km / man_wells if man_wells > 0 else 0
                    wg_km_well = wellgo_km / wellgo_wells if wellgo_wells > 0 else 0
                    
                    # 3. Metrik Head-to-Head
                    delta_km_well = man_km_well - wg_km_well
                    pct_save = (delta_km_well / man_km_well * 100) if man_km_well > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>Total Jarak (Manual)</div><div style='font-size:24px;font-weight:700;color:#E67E22;'>{manual_km:.1f} km</div><div style='font-size:12px;color:#7F8C8D;'>{man_wells} sumur ({man_km_well:.2f} km/well)</div></div>", unsafe_allow_html=True)
                    col2.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>Total Jarak (WELLGO)</div><div style='font-size:24px;font-weight:700;color:{ui.TEAL};'>{wellgo_km:.1f} km</div><div style='font-size:12px;color:#7F8C8D;'>{wellgo_wells} sumur ({wg_km_well:.2f} km/well)</div></div>", unsafe_allow_html=True)
                    
                    if pct_save > 0:
                        col3.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;border: 1px solid {ui.TEAL_GREEN};'><div class='wg-eyb'>Efisiensi KM/Well Ditemukan!</div><div style='font-size:24px;font-weight:700;color:{ui.TEAL_GREEN};'>↓ {pct_save:.1f}%</div><div style='font-size:12px;color:#7F8C8D;'>Menghemat {delta_km_well:.2f} km/well rata-rata armada</div></div>", unsafe_allow_html=True)
                    else:
                        col3.markdown(f"<div class='wg-card' style='padding:15px;text-align:center;'><div class='wg-eyb'>Perbandingan Efisiensi</div><div style='font-size:24px;font-weight:700;color:#E74C3C;'>↑ {abs(pct_save):.1f}%</div><div style='font-size:12px;color:#7F8C8D;'>WELLGO lebih boros {abs(delta_km_well):.2f} km/well</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    comp_search_q = st.text_input("🔎 Pencarian Cepat Nama Sumur (Peta Komparasi)", placeholder="Contoh: BO083", key="comp_search").strip().upper()
                    comp_search_terms = [t for t in comp_search_q.replace(",", " ").split() if t]
                    
                    # 4. Render Peta Head-to-Head Ber-Tooltip Tinggi
                    def render_comparison_map(df_map, lat_col, lon_col, unit_col, well_col, title):
                        layers = []
                        if not df_map.empty:
                            df_map = df_map.copy()
                            
                            df_map["well"] = df_map[well_col]
                            df_map["plan_unit"] = df_map[unit_col]
                            
                            # FIX BUG REINDEXING
                            _raw_dedup = raw.drop_duplicates("well").set_index("well")
                            if "tipe" not in df_map.columns:
                                df_map["tipe"] = df_map["well"].map(_raw_dedup["tipe"]).fillna("REG")
                            if "min_date" not in df_map.columns:
                                df_map["min_date"] = pd.to_datetime(df_map["well"].map(_raw_dedup["min_date"]))
                            if "max_date" not in df_map.columns:
                                df_map["max_date"] = pd.to_datetime(df_map["well"].map(_raw_dedup["max_date"]))
                                
                            if "SCHEDULE DATE" in df_map.columns:
                                df_map["tgl_str"] = pd.to_datetime(df_map["SCHEDULE DATE"]).dt.strftime("%Y-%m-%d").fillna("-")
                            else:
                                df_map["tgl_str"] = pd.to_datetime(df_map.get("plan_day")).dt.strftime("%Y-%m-%d").fillna("-")
                            
                            df_map["min_str"] = pd.to_datetime(df_map["min_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
                            df_map["max_str"] = pd.to_datetime(df_map["max_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
                            df_map["ket"] = df_map["timing_label"].fillna("-") if "timing_label" in df_map.columns else "-"

                            ulabels = sorted(df_map[unit_col].dropna().unique())
                            df_map["color"] = df_map[unit_col].apply(lambda k: cmap(k, ulabels))
                            
                            df_map["hit"] = df_map["well"].str.upper().isin(comp_search_terms) if comp_search_terms else False
                            TIPE_RING = {"NW": [220, 30, 30], "AWS": [245, 150, 20], "REG": [120, 120, 120]}
                            df_map["ring"] = df_map.apply(lambda r: [255, 235, 0] if r["hit"] else TIPE_RING.get(r.get("tipe", "REG"), [120, 120, 120]), axis=1)
                            df_map["ringw"] = np.where(df_map["hit"], 6, np.where(df_map["tipe"].isin(["NW", "AWS"]), 3, 0))

                            if show_block:
                                polys = [{"polygon": block_polygon(sub.rename(columns={lat_col: "lat", lon_col: "lon"})), "color": list(sub["color"].iloc[0]) + [55]} 
                                         for u, sub in df_map.groupby(unit_col) if len(sub) >= 3]
                                if polys: 
                                    layers.append(pdk.Layer("PolygonLayer", data=polys, get_polygon="polygon", get_fill_color="color", get_line_color="color", line_width_min_pixels=1, stroked=True, filled=True))

                            lines = []
                            for u, sub in df_map.groupby(unit_col):
                                s = sub.reset_index(drop=True)
                                if len(s) > 1:
                                    order, _ = optimize_route(s[lat_col].values, s[lon_col].values)
                                    col = list(s["color"].iloc[0])
                                    for a in range(len(order) - 1):
                                        i, j = order[a], order[a + 1]
                                        lines.append({
                                            "from": [s.loc[i, lon_col], s.loc[i, lat_col]], 
                                            "to": [s.loc[j, lon_col], s.loc[j, lat_col]], 
                                            "color": col
                                        })
                            if lines:
                                layers.append(pdk.Layer(
                                    "LineLayer", data=pd.DataFrame(lines), get_source_position="from", 
                                    get_target_position="to", get_color="color", get_width=2
                                ))

                            layers.append(pdk.Layer(
                                "ScatterplotLayer", data=df_map, get_position=[lon_col, lat_col],
                                get_fill_color="color", get_radius=150, get_line_color="ring", get_line_width="ringw",
                                line_width_min_pixels=1, stroked=True, filled=True, opacity=0.8, pickable=True
                            ))
                            
                            # 4. Layer Highlight Pencarian Sumur (Kuning Tebal)
                            hits = df_map[df_map["hit"]].copy()
                            if not hits.empty:
                                layers.append(pdk.Layer(
                                    "ScatterplotLayer", data=hits, get_position=[lon_col, lat_col],
                                    get_fill_color=[255, 215, 0], get_radius=170, get_line_color=[40, 40, 40], get_line_width=5,
                                    line_width_min_pixels=2, stroked=True, filled=True, pickable=True, opacity=0.95
                                ))
                                layers.append(pdk.Layer(
                                    "TextLayer", data=hits, get_position=[lon_col, lat_col], get_text="well",
                                    get_size=75, get_color=[0, 0, 0], get_pixel_offset=[0, -45], # <--- UBAH JADI HITAM [0, 0, 0] DI SINI
                                    font_family="Inter", font_weight="bold", pickable=False
                                ))
                                
                        lat_init = df_map[lat_col].mean() if len(df_map) else 1.6
                        lon_init = df_map[lon_col].mean() if len(df_map) else 101.3
                        
                        tip = "{well} [{tipe}] · {ket}\nTanggal Plan: {tgl_str} | Unit: {plan_unit}\nWindow Execution: {min_str} → {max_str}"
                        view = pdk.ViewState(latitude=lat_init, longitude=lon_init, zoom=8.5)
                        st.caption(f"**{title}**")
                        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="road", tooltip={"text": tip}))

                    map1, map2 = st.columns(2)
                    with map1:
                        render_comparison_map(man_valid, "LAT", "LON", "UNIT", "WELL", "🗺️ Rute Manual (Spaghetti)")
                    with map2:
                        wg_valid = comp_disp[comp_disp["has_coord"]].copy() if len(comp_disp) else pd.DataFrame()
                        render_comparison_map(wg_valid, "lat", "lon", "plan_unit", "well", "🗺️ Rute WELLGO (Optimized)")
                        
        except Exception as e:
            st.error(f"Gagal memproses file manual: {str(e)}. Pastikan format kolom sesuai (WELL, UNIT, SCHEDULE DATE).")

with tab_priority:
    # ── Sumur Prioritas: NW / AWS / PRQ / ORQ ──────────────────────────────
    ui.section("Sumur Prioritas (NW / AWS / PRQ / ORQ)", eyebrow="Pantau kategori prioritas, fase AWS, & window")
    _rt_raw = raw["req_tag"].fillna("") if "req_tag" in raw.columns else pd.Series("", index=raw.index)
    pri_mask = raw["is_nwaws"].fillna(False) | _rt_raw.isin(["PRQ", "ORQ"])
    pri = raw[pri_mask].copy()
    if not len(pri):
        st.info("Tidak ada sumur kategori NW/AWS/PRQ/ORQ pada data ini.")
    else:
        _sched_unit = dict(zip(scheduled_all["well"], scheduled_all["plan_unit"])) if len(scheduled_all) else {}
        _sched_day = dict(zip(scheduled_all["well"], scheduled_all["day_idx"])) if len(scheduled_all) else {}
        _miss_w = set(missed["well"]) if len(missed) else set()
        _left_w = set(leftover["well"]) if len(leftover) else set()

        def _kat(r):
            tp = r["tipe"]; rt = r.get("req_tag", "")
            if tp == "NW": return "NW"
            if tp == "AWS": return "AWS"
            if rt == "PRQ": return "PRQ"
            if rt == "ORQ": return "ORQ"
            return "RTN"

        def _stat(w):
            if w in executed: return "✅ COMP"
            if w in pending_set: return "⏳ PENDING"
            if w in _sched_unit: return "📅 Terjadwal"
            if w in _miss_w: return "⚠️ Miss Deadline"
            if w in _left_w: return "🕓 Antre"
            return "➖ Luar window/exclude"

        pri["Kategori"] = [_kat(r) for _, r in pri.iterrows()]
        pri["Status"] = [_stat(w) for w in pri["well"]]
        cats = ["NW", "AWS", "PRQ", "ORQ"]
        pick = st.multiselect("Filter kategori", cats, default=cats, key="pri_cat")
        view = pri[pri["Kategori"].isin(pick)].copy()
        _cc = view["Kategori"].value_counts()
        st.caption(" · ".join(f"**{k}**: {int(_cc.get(k, 0))}" for k in cats) + f"  ·  total: **{len(view)}**")
        if len(view):
            _onoff = view["status"].apply(lambda s: "🔴 OFF" if str(s).upper().strip() == "OFF" else "🟢 ON") if "status" in view.columns else "🟢 ON"
            pri_disp = view[["well", "Kategori", "category", "field", "area", "min_date", "max_date", "Status"]].copy()
            pri_disp.insert(2, "ON/OFF", _onoff)
            pri_disp["Unit"] = view["well"].map(_sched_unit).fillna("-")
            pri_disp["Hari"] = view["well"].map(_sched_day).apply(lambda x: f"Hari {int(x)}" if pd.notna(x) else "-")
            pri_disp["min_date"] = pd.to_datetime(pri_disp["min_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            pri_disp["max_date"] = pd.to_datetime(pri_disp["max_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            pri_disp = pri_disp.rename(columns={"well": "Well", "category": "Test (sub-kat)", "field": "Field",
                                        "area": "Area", "min_date": "Min Date", "max_date": "Max Date"})
            st.dataframe(pri_disp.sort_values(["Kategori", "Max Date", "Well"]), use_container_width=True, hide_index=True)
            if (view["Kategori"] == "AWS").any():
                st.caption("ℹ️ Kolom **Test (sub-kat)** menampilkan fase AWS (mis. AWS1/AWS2). "
                           "AWS1 (POP+1..+3) dijadwalkan lebih dulu; AWS2 (POP+5..+10) muncul/antre di window-nya sendiri.")

    # ── Wells OFF (untuk verifikasi status ke tim lapangan) ────────────────
    st.divider()
    ui.section("🔌 Wells OFF — Verifikasi Status ke Tim Lapangan", eyebrow="Sumur berstatus OFF di data kandidat")
    off_all = raw[raw["well"].isin(master_off_wells)].copy()
    if not len(off_all):
        st.info("Tidak ada sumur berstatus OFF pada data kandidat.")
    else:
        off_disp = off_all[["well", "field", "area", "subarea", "category", "unit"]].copy()
        off_disp.insert(6, "Di-skip (NCMP+OFF)", off_all["well"].isin(woff_set).map({True: "ya", False: "-"}))
        off_disp = off_disp.rename(columns={"well": "Well", "field": "Field", "area": "Area",
                                            "subarea": "Sub-area", "category": "Kategori", "unit": "Unit Terakhir"})
        st.caption(f"**{len(off_disp)}** sumur berstatus OFF di **master data** (semua, lintas window) — "
                   "verifikasi ON/OFF aktual ke tim lapangan. Sumur OFF di-skip dari penjadwalan; "
                   "bila ternyata ON, ubah **Well Status** di Excel kandidat lalu re-run.")
        st.caption(f"ℹ️ Berbeda dgn KPI **WELLS OFF ({len(off_wells)})** di dashboard yang hanya menghitung OFF "
                   "yang jadi kandidat **di siklus ini** (dalam window & belum COMP/PENDING).")
        st.dataframe(off_disp.sort_values(["Area", "Well"]), use_container_width=True, hide_index=True)

    # ── Paksa Eligible: NW/AWS status OFF → ON ─────────────────────────────
    st.divider()
    ui.section("✅ Paksa Eligible — NW/AWS Status OFF", eyebrow="Override OFF→ON agar masuk eligible & dijadwalkan")
    if not len(nwaws_off_pool):
        st.info("Tidak ada sumur NW/AWS berstatus OFF di data kandidat.")
    else:
        st.caption("Status ON/OFF di Excel kemungkinan **belum terupdate**. Pilih sumur **NW/AWS** yang sebenarnya "
                   "sudah ON agar di-override jadi eligible & dijadwalkan untuk siklus ini "
                   "(tetap tunduk pada window min–max-nya).")
        _lbl = {f"{r['well']}  —  {r['tipe']} ({_fv(r['category'])}) / {_fv(r['field'])}": r['well']
                for _, r in nwaws_off_pool.sort_values(["tipe", "well"]).iterrows()}
        _cur = [l for l, w in _lbl.items() if w in force_on]
        _pick = st.multiselect("Sumur NW/AWS OFF → paksa ON:", list(_lbl.keys()), default=_cur, key="force_on_ms")
        _picked = [_lbl[l] for l in _pick]
        if set(_picked) != force_on:
            st.session_state["force_on_nwaws"] = _picked
            st.rerun()
        if force_on:
            st.caption(f"✅ **{len(force_on)}** sumur NW/AWS dipaksa ON & masuk eligible: {', '.join(sorted(force_on))}.")

# ── Footer Cleanup Action Trigger Module ────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns([4, 1])
with col_f1:
    st.caption("WELLGO (Well Grouping Optimizer). Dikembangkan oleh Tim Well Test SL North")
with col_f2:
    if st.button("🔄 Hard Reset Konfigurasi", use_container_width=True):
        st.session_state["manual_assign"] = {}
        st.session_state["manual_unassign"] = []
        st.session_state["field_assign"] = {}
        st.session_state["manual_comp"] = []
        st.session_state["force_on_nwaws"] = []
        st.session_state["rescued_wells"] = []
        st.rerun()