"""
WELL TEST GROUPING OPTIMIZER  (H-1 daily planner)
-------------------------------------------------
9 unit tes (MWT / MPAS_xxx). Mode dedicated (per territory) / pooled (unit bebas).
Kriteria optimasi: jarak saja / +durasi / +min-max / +durasi+min-max.
Visual block area per grup. Exclude area tertentu (default LIBO).

Run:  python -m streamlit run app.py
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

st.set_page_config(page_title="Well Test Grouping Optimizer", layout="wide")
DB_PATH = "welltest_status.db"
SHEET_DEFAULT = "Kandidat Sumur"

# Zona standby unit (mode pooled): unit gak bisa lintas zona
REMOTE_AREAS = {"BANGKO", "BALAM"}                                  # remote
REMOTE_UNITS = ["MPAS_444", "MPAS_768", "MPAS_523", "MPAS_445", "MPAS_534"]
NONREMOTE_UNITS = ["MPAS_535", "MPAS_524", "MPAS_525", "MPAS_767"]  # non-remote (BEKASAP)


# ------------------------------------------------------------------ persistence
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS execution_log(
        plan_date TEXT, well_name TEXT, unit TEXT, status TEXT, reason TEXT, updated_at TEXT,
        PRIMARY KEY(plan_date, well_name))""")
    # migrasi DB lama: tambah kolom yg belum ada (mis. dari skema versi sebelumnya)
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
    """Status COMP/NCMP per well dalam PERIODE [lo..hi], ambil record TERBARU per well.
       (NCMP yg belakangan jadi COMP -> kebaca COMP). Record di luar periode diabaikan."""
    con = sqlite3.connect(DB_PATH)
    try:
        q = ("SELECT well_name AS well, status, reason, plan_date FROM execution_log "
             "WHERE status IN ('executed','ncmp') AND plan_date BETWEEN ? AND ?")
        df = pd.read_sql(q, con, params=(str(lo), str(hi)))
    except Exception:
        df = pd.DataFrame(columns=["well", "status", "reason", "plan_date"])
    con.close()
    if len(df):
        df = df.sort_values("plan_date").groupby("well", as_index=False).last()
    executed = set(df.loc[df["status"] == "executed", "well"])
    ncmp = df[df["status"] == "ncmp"][["well", "reason", "plan_date"]].copy()
    return executed, ncmp


def norm_unit(u):
    """MP444 -> MPAS_444 ; biarkan unit TS apa adanya."""
    u = str(u).strip().upper()
    m = re.fullmatch(r"MP_?(\d+)", u)
    return f"MPAS_{m.group(1)}" if m else u


def import_compncmp(file_list):
    """Baca file COMP/NCMP harian -> update execution_log.
       COMP=executed (keluar dari pool), NCMP=ncmp (+alasan) -> dijadwalkan ulang."""
    n_comp = n_ncmp = 0
    reasons = {}
    con = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat(timespec="seconds")
    for fb in file_list:
        xls = pd.ExcelFile(BytesIO(fb))
        sht = next((s for s in xls.sheet_names if s.strip().upper().replace(" ", "")
                    in ("SCHDATABASE", "COMPNCMP", "SCHSTATUS")), xls.sheet_names[0])
        df = pd.read_excel(xls, sheet_name=sht)
        cols = {str(c).strip().upper(): c for c in df.columns}
        cw = cols.get("WELL")
        cs = cols.get("STATUS")
        cd = cols.get("SCHEDULE_DATE_TEST")
        cu = cols.get("UNIT")
        cr = cols.get("COMMENT IF NOT COMPLETE")
        if not (cw and cs and cd):
            continue
        for _, r in df.iterrows():
            well = str(r[cw]).strip()
            stat = str(r[cs]).strip().upper()
            if well in ("", "nan") or stat not in ("COMP", "NCMP"):
                continue
            try:
                pdate = pd.to_datetime(r[cd]).date().isoformat()
            except Exception:
                continue
            unit = norm_unit(r[cu]) if cu else ""
            reason = (str(r[cr]).strip().upper() if cr and pd.notna(r[cr]) else "")
            log_status = "executed" if stat == "COMP" else "ncmp"
            con.execute("""INSERT INTO execution_log(plan_date,well_name,unit,status,reason,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(plan_date,well_name) DO UPDATE SET
                unit=excluded.unit, status=excluded.status, reason=excluded.reason,
                updated_at=excluded.updated_at""", (pdate, well, unit, log_status, reason, now))
            if stat == "COMP":
                n_comp += 1
            else:
                n_ncmp += 1
                reasons[reason or "(kosong)"] = reasons.get(reason or "(kosong)", 0) + 1
    con.commit()
    con.close()
    return {"comp": n_comp, "ncmp": n_ncmp, "reasons": reasons}



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
    """Parse tanggal: datetime/string biasa ATAU serial Excel (mis. 46186 = 2026-06-18)."""
    num = pd.to_numeric(col, errors="coerce")
    valid = num.dropna()
    # kalau mayoritas angka & ada di rentang serial Excel (~1954-2064) -> serial Excel
    if len(valid) and valid.between(20000, 60000).mean() > 0.5:
        return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(col, errors="coerce")


@st.cache_data(show_spinner=False)
def load_candidates(file_bytes, sheet):
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "well_name": "well", "Surface Lat": "lat", "Surface Lon": "lon",
        "Duration test (minutes)": "dur", "min_execution date": "min_date",
        "max_execution_date": "max_date", "op_sub_area_code": "subarea",
        "op_area_code": "area", "test_category": "category", "unit_name": "unit",
        "well_tier": "tier", "field": "field"})
    for c in ["lat", "lon"]:
        if c not in df.columns:
            df[c] = np.nan
    df["min_date"] = to_dt(df["min_date"])
    df["max_date"] = to_dt(df["max_date"])
    df["is_mpas"] = df["unit"].astype(str).str.upper().str.startswith("MPAS")

    # Well Status (ON/OFF) — prioritas kolom "Well Status", fallback "last_status"
    status_col = next((c for c in df.columns if c.strip().upper() in ("WELL STATUS", "LAST_STATUS")), None)
    df["status"] = (df[status_col].astype(str).str.upper().str.strip() if status_col else "ON")

    # SCH Status (COMP/NCMP/blank) langsung dari kolom kandidat
    sch_col = next((c for c in df.columns if c.strip().upper() in ("SCH STATUS", "SCH_STATUS")), None)
    df["sch_status"] = (df[sch_col].astype(str).str.upper().str.strip() if sch_col else "")
    df["sch_status"] = df["sch_status"].replace({"NAN": "", "NONE": ""})
    return df


def good_coord(lat, lon):
    return pd.notna(lat) & pd.notna(lon) & lat.between(0.1, 5) & lon.between(95, 110)


def resolve_coords(df, cache):
    df = df.copy()
    df["coord_source"] = np.where(good_coord(df["lat"], df["lon"]), "database", None)
    if not cache.empty:
        cmap = cache.set_index("well_name")
        miss = df["coord_source"].isna() & df["well"].isin(cmap.index)
        df.loc[miss, "lat"] = df.loc[miss, "well"].map(cmap["lat"])
        df.loc[miss, "lon"] = df.loc[miss, "well"].map(cmap["lon"])
        df.loc[miss, "coord_source"] = "cache"
    base = df[df["coord_source"].isin(["database", "cache"])]
    for key in ["field", "subarea", "area"]:
        cent = base.groupby(key)[["lat", "lon"]].mean()
        miss = df["coord_source"].isna() & df[key].isin(cent.index)
        df.loc[miss, "lat"] = df.loc[miss, key].map(cent["lat"])
        df.loc[miss, "lon"] = df.loc[miss, key].map(cent["lon"])
        df.loc[miss, "coord_source"] = f"imputed_{key}"
    df["coord_source"] = df["coord_source"].fillna("none")
    df["has_coord"] = df["coord_source"] != "none"
    return df


# ------------------------------------------------------------------ geometry
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def nn_route(lat, lon):
    n = len(lat)
    if n <= 1:
        return list(range(n)), 0.0
    order, used, total = [0], {0}, 0.0
    for _ in range(n - 1):
        c = order[-1]
        best, bd = None, 1e18
        for j in range(n):
            if j in used:
                continue
            d = haversine_km(lat[c], lon[c], lat[j], lon[j])
            if d < bd:
                bd, best = d, j
        order.append(best)
        used.add(best)
        total += bd
    return order, total


def convex_hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def block_polygon(sub, pad_km=0.6):
    """Polygon 'block area' utk satu grup: convex hull (>=3 titik) atau lingkaran."""
    lat = sub["lat"].values
    lon = sub["lon"].values
    clat, clon = lat.mean(), lon.mean()
    if len(sub) >= 3:
        hull = convex_hull(list(zip(lon, lat)))
        if len(hull) >= 3:
            f = 1.0 + pad_km / max(0.3, np.mean(haversine_km(lat, lon, clat, clon)) + 0.3)
            return [[clon + (x - clon) * f, clat + (y - clat) * f] for x, y in hull]
    r = max(haversine_km(lat, lon, clat, clon).max() if len(sub) > 1 else 0.0, 0.0) + pad_km
    out = []
    for k in range(28):
        a = 2 * math.pi * k / 28
        dlat = (r / 111.0) * math.sin(a)
        dlon = (r / (111.0 * math.cos(math.radians(clat)))) * math.cos(a)
        out.append([clon + dlon, clat + dlat])
    return out


# ------------------------------------------------------------------ engine
def grow_group(idxs, used, lat, lon, dur, max_wells, time_budget, speed, use_dur, seed):
    members = [seed]
    used.add(seed)
    t = dur[seed]
    cur = seed
    while len(members) < max_wells:
        rem = [i for i in idxs if i not in used]
        if not rem:
            break
        d = haversine_km(lat[cur], lon[cur], lat[rem], lon[rem])
        k = rem[int(np.argmin(d))]
        travel = (float(np.min(d)) / speed) * 60.0
        if use_dur and t + travel + dur[k] > time_budget:
            break
        members.append(k)
        used.add(k)
        t += travel + dur[k]
        cur = k
    return members


def plan(elig, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur):
    """mode: 'dedicated' (per unit) / 'pooled' (unit bebas).
       use_urg: pertimbangkan min-max (deadline). use_dur: pertimbangkan durasi (time budget)."""
    df = elig.reset_index(drop=True).copy()
    df["scheduled"] = False
    lat, lon, dur = df["lat"].values, df["lon"].values, df["dur"].values
    urg = df["urgency"].values

    if mode == "dedicated":
        for unit in df["unit"].dropna().unique():
            idxs = list(df.index[df["unit"] == unit])
            if not idxs:
                continue
            if use_urg:
                ordered = sorted(idxs, key=lambda i: (urg[i], dur[i]))
                sel = ordered[:max_wells]
                if use_dur:
                    while len(sel) > 1:
                        sl, ln = lat[sel], lon[sel]
                        _, dist = nn_route(sl, ln)
                        if df.loc[sel, "dur"].sum() + (dist / speed) * 60 <= time_budget:
                            break
                        sel = sorted(sel, key=lambda i: urg[i])[:-1]  # buang paling tdk urgent
            else:
                # seed = paling sentral (min total jarak ke sesama unit)
                if len(idxs) == 1:
                    seed = idxs[0]
                else:
                    tot = [haversine_km(lat[i], lon[i], lat[idxs], lon[idxs]).sum() for i in idxs]
                    seed = idxs[int(np.argmin(tot))]
                sel = grow_group(idxs, set(), lat, lon, dur, max_wells, time_budget, speed, use_dur, seed)
            df.loc[sel, "scheduled"] = True
        df["plan_unit"] = df["unit"]
    else:  # pooled — unit bebas TAPI dibatasi zona remote/non-remote
        df["plan_unit"] = None

        def pool_zone(zone_idx, k, labels):
            if not zone_idx or k <= 0:
                return
            sub = df.loc[zone_idx]
            if use_urg:
                seed_order = list(sub.sort_values(["urgency", "dur"]).index)
            else:
                seed_order = list(sub.sort_values(["lon", "lat"]).index)  # sweep barat->timur
            used, groups = set(), []
            for s in seed_order:
                if len(groups) >= k:
                    break
                if s in used:
                    continue
                members = grow_group(zone_idx, used, lat, lon, dur, max_wells,
                                     time_budget, speed, use_dur, s)
                groups.append(members)
            for gi, members in enumerate(groups):
                lbl = labels[gi] if gi < len(labels) else f"{labels[0]}+{gi}"
                df.loc[members, "scheduled"] = True
                df.loc[members, "plan_unit"] = lbl

        remote_idx = list(df.index[df["area"].isin(REMOTE_AREAS)])
        nonremote_idx = list(df.index[~df["area"].isin(REMOTE_AREAS)])
        pool_zone(remote_idx, n_remote, REMOTE_UNITS)
        pool_zone(nonremote_idx, n_nonremote, NONREMOTE_UNITS)
    return df


def plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur):
    """Rollout harian: tiap hari jadwalkan dari sumur yg belum ke-jadwal & window-nya buka.
       Yg sudah ke-jadwal hari-N gak masuk pool hari N+1."""
    elig = elig.reset_index(drop=True).copy()
    elig["scheduled"] = False
    elig["plan_unit"] = None
    elig["plan_day"] = pd.NaT
    elig["day_idx"] = 0
    rem = pd.Series(True, index=elig.index)
    for i, day in enumerate(days, start=1):
        pidx = elig.index[rem & (elig["min_date"] <= day) & (elig["max_date"] >= day)]
        if len(pidx) == 0:
            continue
        pool = elig.loc[pidx].copy()
        pool["urgency"] = (pool["max_date"] - day).dt.days
        pd_ = plan(pool, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur)
        sd = pd_[pd_["scheduled"]]
        if len(sd) == 0:
            continue
        sidx = elig.index[elig["well"].isin(sd["well"])]
        elig.loc[sidx, "scheduled"] = True
        elig.loc[sidx, "plan_day"] = day
        elig.loc[sidx, "day_idx"] = i
        elig.loc[sidx, "plan_unit"] = elig.loc[sidx, "well"].map(dict(zip(sd["well"], sd["plan_unit"])))
        rem.loc[sidx] = False
    elig["urgency"] = (elig["max_date"] - days[0]).dt.days
    return elig


def unit_summary(df, speed):
    rows = []
    for unit, sub in df[df["scheduled"]].groupby("plan_unit"):
        c = sub[sub["has_coord"]]
        dist = nn_route(c["lat"].values, c["lon"].values)[1] if len(c) > 1 else 0.0
        rows.append({
            "Unit": unit, "Sumur": len(sub), "Test (min)": int(sub["dur"].sum()),
            "Rute (km)": round(dist, 1), "Est (min)": int(sub["dur"].sum() + (dist / speed) * 60),
            "Sub-area": ", ".join(sorted(sub["subarea"].dropna().unique())),
            "Deadline tercepat": sub["max_date"].min().date(),
            "Wells": ", ".join(sub["well"])})
    return pd.DataFrame(rows).sort_values("Unit")


COLORS = [[228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0],
          [166, 86, 40], [247, 129, 191], [26, 188, 156], [241, 196, 15], [106, 61, 154],
          [178, 223, 138], [251, 154, 153]]


def cmap(label, labels):
    try:
        return COLORS[list(labels).index(label) % len(COLORS)]
    except ValueError:
        return [130, 130, 130]


# ================================================================== UI
init_db()
st.title("🛢️ Well Test Grouping Optimizer — 9 Unit MWT")

CRIT = {
    "Kedekatan jarak saja": (False, False),
    "Jarak + durasi test": (False, True),
    "Jarak + min-max (deadline)": (True, False),
    "Jarak + durasi + min-max": (True, True),
}

with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Upload Excel kandidat", type=["xlsx"])
    sheet = st.text_input("Nama sheet", SHEET_DEFAULT)
    mpas_only = st.checkbox("Hanya Unit Tes (MPAS), exclude TS", value=True)
    st.markdown("**Status eksekusi (COMP/NCMP)**")
    comp_files = st.file_uploader("Upload file COMP/NCMP harian (boleh banyak)",
                                  type=["xlsx"], accept_multiple_files=True)
    skip_woff = st.checkbox("Skip well NCMP-WOFF dari penjadwalan ulang", value=True,
                            help="WOFF = well lagi off, gak bisa dites; jangan dijadwalin ulang dulu")
    _today = datetime.now().date()
    periode = st.date_input("Periode siklus (baca COMP/NCMP rentang ini saja)",
                            value=(_today, _today + timedelta(days=6)),
                            help="NCMP/COMP hanya dibaca dalam rentang ini — hindari ketarik record lama (mis. 2025)")

    st.header("2. Mode & kriteria")
    mode_label = st.radio("Mode unit", ["Dedicated (unit per territory)",
                                        "Pooled (unit bebas, murni kedekatan)"])
    mode = "dedicated" if mode_label.startswith("Dedicated") else "pooled"
    crit_label = st.radio("Kriteria optimasi", list(CRIT.keys()), index=3)
    use_urg, use_dur = CRIT[crit_label]

    st.header("3. Kapasitas")
    target = st.date_input("Tanggal mulai (hari ke-1)", datetime.now().date() + timedelta(days=1))
    horizon = st.slider("Horizon planning (hari)", 1, 7, 7)
    max_wells = st.slider("Sumur / unit / hari", 3, 8, 6)
    ded = (mode == "dedicated")
    n_remote = st.slider("Unit remote (BANGKO/BALAM)", 1, 5, 5, disabled=ded)
    n_nonremote = st.slider("Unit non-remote (BEKASAP)", 1, 4, 4, disabled=ded)
    time_budget = st.slider("Time budget / unit (menit)", 180, 540, 360, 30, disabled=not use_dur)
    speed = st.slider("Kecepatan unit (km/jam)", 10, 60, 25, 5)

    st.header("4. Visual")
    show_block = st.checkbox("Tampilkan block area per grup", value=True)

if up is None:
    st.info("⬅️ Upload file Excel kandidat (sheet `Kandidat Sumur`) buat mulai.")
    st.stop()

raw = load_candidates(up.getvalue(), sheet)

# import status COMP/NCMP (kalau ada) ----------------------------------------
if comp_files:
    summ_imp = import_compncmp([f.getvalue() for f in comp_files])
    rtxt = ", ".join(f"{k}: {v}" for k, v in summ_imp["reasons"].items()) or "-"
    st.success(f"✅ Import status: {summ_imp['comp']} COMP (executed), "
               f"{summ_imp['ncmp']} NCMP (dijadwalkan ulang). Alasan NCMP → {rtxt}")
if mpas_only:
    raw = raw[raw["is_mpas"]].copy()

with st.sidebar:
    all_areas = sorted(raw["area"].dropna().unique())
    default_excl = [a for a in all_areas if a == "LIBO"]
    excl_areas = st.multiselect("Exclude area", all_areas, default=default_excl)
if excl_areas:
    raw = raw[~raw["area"].isin(excl_areas)].copy()

raw = resolve_coords(raw, load_coord_cache())
target_ts = pd.Timestamp(target)

# eligibility (level minggu) ---------------------------------------------------
days = [target_ts + pd.Timedelta(days=i) for i in range(horizon)]
# periode siklus (buat baca COMP/NCMP) — handle date_input bisa balik 1 atau 2 tanggal
if isinstance(periode, (list, tuple)) and len(periode) == 2:
    per_lo, per_hi = periode[0], periode[1]
else:
    per_lo = periode if not isinstance(periode, (list, tuple)) else periode[0]
    per_hi = per_lo + timedelta(days=6)
executed_log, ncmp_log = status_in_period(per_lo, per_hi)
comp_col = set(raw.loc[raw["sch_status"] == "COMP", "well"])       # COMP dari kolom SCH Status
executed = executed_log | comp_col                                # gabung kolom + log uploader
ncmp_log = ncmp_log[~ncmp_log["well"].isin(executed)].copy()
ncmp_col = set(raw.loc[raw["sch_status"] == "NCMP", "well"]) - executed
ncmp_set = (set(ncmp_log["well"]) | ncmp_col) - executed           # semua NCMP dalam periode
woff_set = set(ncmp_log.loc[ncmp_log["reason"] == "WOFF", "well"]) if skip_woff else set()
ncmp_set -= woff_set                                              # WOFF di-skip dari reschedule

in_raw = set(raw["well"])
ncmp_replan = ncmp_set & in_raw                                    # ada di Excel kandidat -> bisa diplan ulang
ncmp_no_data = sorted(ncmp_set - in_raw)                           # NCMP tapi gak ada di Excel kandidat

# gabungan info NCMP buat panel (reason dari log, kolom = "(kolom)")
ncmp_col_df = pd.DataFrame({"well": sorted(ncmp_col), "reason": "", "plan_date": "(kolom)"})
ncmp_df = pd.concat([ncmp_log, ncmp_col_df], ignore_index=True).drop_duplicates("well")

week_lo, week_hi = days[0], days[-1]
# kandidat = window overlap horizon, ATAU NCMP carry-over (overdue, wajib dijadwalkan ulang)
overlap = (raw["min_date"] <= week_hi) & (raw["max_date"] >= week_lo)
is_ncmp = raw["well"].isin(ncmp_replan)
comp_wells = raw[overlap & raw["well"].isin(executed)].copy()      # sudah selesai siklus ini
cand = raw[(overlap | is_ncmp) & (~raw["well"].isin(executed))].copy()
off_wells = cand[cand["status"] == "OFF"].copy()                   # Well Status OFF -> tdk diplanning
woff_wells = raw[raw["well"].isin(woff_set)].copy()                # NCMP-WOFF -> skip reschedule
elig_all = cand[(cand["status"] != "OFF") & (~cand["well"].isin(woff_set))].copy()
elig_all["carry_ncmp"] = elig_all["well"].isin(ncmp_replan)        # tandai carry-over NCMP
elig_all["urgency"] = (elig_all["max_date"] - week_lo).dt.days     # NCMP overdue -> urgency negatif (prioritas)
elig = elig_all[elig_all["has_coord"]].copy()
nocoord = elig_all[~elig_all["has_coord"]].copy()

# rollout 7 hari -------------------------------------------------------------
if len(elig):
    week_df = plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur)
else:
    week_df = elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)

week_df["zone"] = np.where(week_df["area"].isin(REMOTE_AREAS), "remote", "non-remote")
scheduled_all = week_df[week_df["scheduled"]].copy()
sched_wells = set(scheduled_all["well"])
leftover = week_df[~week_df["scheduled"]].copy()                    # gak ke-jadwal sepanjang minggu
missed = leftover[leftover["max_date"] <= week_hi]                  # deadline lewat dalam horizon = risiko

zone_note = " | Pooled dibatasi zona remote/non-remote" if mode == "pooled" else ""
st.caption(f"Mode: **{mode_label}** | Kriteria: **{crit_label}** | "
           f"Horizon: **{horizon} hari** ({week_lo.date()} → {week_hi.date()}) | "
           f"Periode COMP/NCMP: **{per_lo} → {per_hi}** | Exclude: {excl_areas or '-'}{zone_note}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Eligible (minggu)", len(elig_all))
c2.metric("Terjadwal (minggu)", len(scheduled_all))
c3.metric("Belum terjadwal", len(leftover))
c4.metric("⚠️ Miss deadline", len(missed))
c5.metric("Tanpa koordinat", len(nocoord))
c6.metric("🔌 Well OFF", len(off_wells))

if len(scheduled_all) == 0 and len(nocoord) == 0:
    st.warning("Gak ada sumur eligible/ter-mapping di rentang minggu ini.")
    st.stop()

# pilih tampilan: toggle semua / per hari ------------------------------------
day_opts = [f"📅 Semua ({horizon} hari)"] + [f"Hari {i+1}" for i in range(horizon)]
if horizon > 1:
    pick = st.radio("Tampilkan grouping untuk", day_opts, horizontal=True, index=0,
                    help="Block area & rute cuma muncul saat 1 hari dipilih")
else:
    pick = day_opts[1]
view_day = None if str(pick).startswith("📅") else days[day_opts.index(pick) - 1]
if view_day is not None:
    st.caption(f"📍 Menampilkan **Hari {day_opts.index(pick)}** — {view_day.date()}")

disp = scheduled_all if view_day is None else scheduled_all[scheduled_all["plan_day"] == view_day]

# overview mingguan ----------------------------------------------------------
if view_day is None:
    ov = (scheduled_all.assign(jam=scheduled_all["dur"] / 60)
          .groupby(["day_idx", "plan_day"])
          .agg(Sumur=("well", "size"), Unit=("plan_unit", "nunique"),
               Jam_test=("jam", "sum")).reset_index())
    ov["Tanggal"] = ov["plan_day"].dt.date
    ov["Jam_test"] = ov["Jam_test"].round(1)
    ov = ov.rename(columns={"day_idx": "Hari"})[["Hari", "Tanggal", "Sumur", "Unit", "Jam_test"]]
    st.subheader("📆 Overview mingguan")
    st.dataframe(ov, use_container_width=True, hide_index=True)

# miss-deadline panel --------------------------------------------------------
if len(missed):
    st.error(f"⚠️ {len(missed)} sumur deadline-nya lewat dalam {horizon} hari ini tapi gak kebagian slot "
             "(kapasitas 9 unit gak cukup). Pertimbangkan tambah shift / unit / perpanjang horizon.")
    st.dataframe(missed[["well", "unit", "subarea", "category", "urgency", "max_date"]]
                 .rename(columns={"max_date": "deadline", "unit": "unit_asli"}).sort_values("urgency"),
                 use_container_width=True, hide_index=True)

# well sudah COMP (selesai siklus ini) ---------------------------------------
if len(comp_wells):
    with st.expander(f"✅ {len(comp_wells)} well COMP — sudah selesai siklus ini (di-exclude)"):
        st.dataframe(comp_wells[["well", "unit", "subarea", "category", "dur", "sch_status"]]
                     .rename(columns={"unit": "unit_asli", "dur": "durasi", "sch_status": "SCH"}),
                     use_container_width=True, hide_index=True)

# well OFF (tidak diplanning) ------------------------------------------------
if len(off_wells):
    with st.expander(f"🔌 {len(off_wells)} well status OFF — TIDAK diplanning (klik buat lihat)"):
        st.dataframe(off_wells[["well", "unit", "subarea", "area",
                                "category", "dur", "max_date", "status"]]
                     .rename(columns={"unit": "unit_asli", "dur": "durasi", "max_date": "deadline"}),
                     use_container_width=True, hide_index=True)

# NCMP carry-over: dijadwalkan ulang vs tidak bisa (gak ada di Excel kandidat) ---
replan_df = ncmp_df[ncmp_df["well"].isin(ncmp_replan)]
if len(replan_df):
    with st.expander(f"🔁 {len(replan_df)} well NCMP — DIJADWALKAN ULANG (masuk eligible, klik lihat)"):
        st.dataframe(replan_df.rename(columns={"plan_date": "tgl_NCMP", "reason": "alasan"}),
                     use_container_width=True, hide_index=True)
if ncmp_no_data:
    no_df = ncmp_df[ncmp_df["well"].isin(ncmp_no_data)]
    st.warning(f"⚠️ {len(ncmp_no_data)} well NCMP TIDAK ada di Excel kandidat → gak bisa dijadwalkan "
               "(tambahkan baris well ini ke Excel kandidat kalau memang perlu dites ulang).")
    with st.expander("Lihat daftar NCMP yang tidak ada di Excel kandidat"):
        st.dataframe(no_df.rename(columns={"plan_date": "tgl_NCMP", "reason": "alasan"}),
                     use_container_width=True, hide_index=True)
if len(woff_wells):
    st.warning(f"⏸️ {len(woff_wells)} well NCMP-WOFF di-skip dari penjadwalan ulang (well lagi off). "
               "Uncheck opsi di sidebar kalau mau tetap dijadwalin.")
    st.dataframe(woff_wells[["well", "unit", "subarea", "category", "max_date"]]
                 .rename(columns={"unit": "unit_asli", "max_date": "deadline"}),
                 use_container_width=True, hide_index=True)

# map ------------------------------------------------------------------------
title = "🗺️ Visual grouping — semua hari" if view_day is None else f"🗺️ Grouping {view_day.date()}"
st.subheader(title)
pmap = disp[disp["has_coord"]].copy()
if len(pmap):
    by_day = view_day is None
    if by_day:
        labels = sorted(pmap["day_idx"].unique())
        pmap["ckey"] = pmap["day_idx"]
    else:
        labels = sorted(pmap["plan_unit"].dropna().unique())
        pmap["ckey"] = pmap["plan_unit"]
    pmap["color"] = pmap["ckey"].apply(lambda k: cmap(k, labels))
    pmap["radius"] = np.where(pmap["coord_source"].str.startswith("imputed"), 90, 170)
    layers = []
    # block area + rute hanya di tampilan single-day (per unit); di "semua" cuma titik per hari
    if not by_day:
        if show_block:
            polys = [{"polygon": block_polygon(sub), "color": cmap(u, labels) + [55]}
                     for u, sub in pmap.groupby("plan_unit")]
            layers.append(pdk.Layer("PolygonLayer", data=polys, get_polygon="polygon",
                get_fill_color="color", get_line_color="color", line_width_min_pixels=1,
                stroked=True, filled=True, pickable=False))
        lines = []
        for u, sub in pmap.groupby("plan_unit"):
            s = sub.reset_index(drop=True)
            order, _ = nn_route(s["lat"].values, s["lon"].values)
            col = cmap(u, labels)
            for a in range(len(order) - 1):
                i, j = order[a], order[a + 1]
                lines.append({"from": [s.loc[i, "lon"], s.loc[i, "lat"]],
                              "to": [s.loc[j, "lon"], s.loc[j, "lat"]], "color": col})
        if lines:
            layers.append(pdk.Layer("LineLayer", data=pd.DataFrame(lines), get_source_position="from",
                get_target_position="to", get_color="color", get_width=2))
    layers.append(pdk.Layer("ScatterplotLayer", data=pmap, get_position=["lon", "lat"],
        get_fill_color="color", get_radius="radius", pickable=True, opacity=0.9))
    view = pdk.ViewState(latitude=float(pmap["lat"].mean()), longitude=float(pmap["lon"].mean()), zoom=8.5)
    tip = "{well}\nHari {day_idx} | {plan_unit} | {dur} menit" if by_day else "{well}\n{plan_unit} | {subarea} | {dur} menit"
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="road",
        tooltip={"text": tip}))
    st.caption("Warna = hari (tampilan semua) / unit (tampilan per hari). Titik kecil = koordinat imputasi.")

# ringkasan per unit (untuk hari terpilih) -----------------------------------
if view_day is not None:
    st.subheader(f"📋 Ringkasan per unit — {view_day.date()}")
    st.dataframe(unit_summary(disp, speed), use_container_width=True, hide_index=True)

# jadwal detail --------------------------------------------------------------
st.subheader("🗂️ Jadwal detail")
scols = ["day_idx", "plan_day", "plan_unit", "zone", "unit", "well", "subarea", "category",
         "dur", "urgency", "max_date", "coord_source"]
det = disp[scols].rename(columns={"day_idx": "hari", "plan_day": "tanggal", "plan_unit": "grup",
                                  "unit": "unit_asli", "dur": "durasi", "max_date": "deadline"})
det["tanggal"] = det["tanggal"].dt.date
st.dataframe(det.sort_values(["hari", "grup", "urgency"]), use_container_width=True, hide_index=True)

# no-coord panel -------------------------------------------------------------
if len(nocoord):
    st.subheader(f"⚠️ Tanpa koordinat — {len(nocoord)} sumur")
    nc = nocoord[["well", "unit", "field", "category", "max_date", "urgency"]].copy()
    nc["lat"] = np.nan
    nc["lon"] = np.nan
    nc_edit = st.data_editor(nc.rename(columns={"max_date": "deadline"}), use_container_width=True,
        height=160, num_rows="fixed", key="nc_editor",
        column_config={"lat": st.column_config.NumberColumn("lat", format="%.6f"),
                       "lon": st.column_config.NumberColumn("lon", format="%.6f")},
        disabled=["well", "unit", "field", "category", "deadline", "urgency"])
    if st.button("💾 Simpan koordinat manual"):
        pairs = [(r["well"], r["lat"], r["lon"]) for _, r in nc_edit.iterrows()
                 if pd.notna(r["lat"]) and pd.notna(r["lon"])]
        save_coords(pairs)
        st.success(f"{len(pairs)} koordinat tersimpan. Refresh buat re-routing.")

# eksekusi (hanya saat satu hari dipilih) + export ---------------------------
st.subheader("✅ Update eksekusi & export")
if view_day is not None:
    st.caption(f"Tandai sumur yg sudah dieksekusi tgl {view_day.date()}. Yg sudah dieksekusi "
               "otomatis keluar dari rollout (gak masuk hari berikutnya) saat refresh.")
    done = st.multiselect("Sumur sudah dieksekusi", sorted(disp["well"]))
    if st.button("💾 Simpan status eksekusi", type="primary"):
        rows = [(r["well"], str(r["plan_unit"]), "executed" if r["well"] in done else "planned")
                for _, r in disp.iterrows()]
        save_status(str(view_day.date()), rows)
        st.success(f"Tersimpan utk {view_day.date()}. {len(done)} dieksekusi. Refresh buat re-plan.")
else:
    st.caption("Pilih satu hari di atas buat update status eksekusi.")

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
exp_cols = ["day_idx", "plan_day", "plan_unit", "zone", "unit", "well", "subarea", "field",
            "category", "dur", "min_date", "max_date", "urgency", "coord_source", "lat", "lon"]
ren = {"day_idx": "hari", "plan_day": "tanggal", "plan_unit": "grup", "unit": "unit_asli",
       "dur": "durasi_test_menit", "max_date": "deadline"}

ex1, ex2 = st.columns(2)

# --- export mingguan ---
out_w = BytesIO()
with pd.ExcelWriter(out_w, engine="openpyxl") as w:
    scheduled_all[exp_cols].rename(columns=ren).sort_values(
        ["hari", "grup", "urgency"]).to_excel(w, sheet_name="Jadwal_Mingguan", index=False)
    if view_day is None:
        ov.to_excel(w, sheet_name="Overview", index=False)
    if len(missed):
        missed[["well", "unit", "subarea", "category", "dur", "urgency", "max_date"]].rename(
            columns={"dur": "durasi_test_menit", "max_date": "deadline", "unit": "unit_asli"}).to_excel(
            w, sheet_name="Miss-Deadline", index=False)
    if len(off_wells):
        off_wells[["well", "unit", "subarea", "category", "dur", "status"]].rename(
            columns={"unit": "unit_asli", "dur": "durasi_test_menit"}).to_excel(
            w, sheet_name="Well-OFF", index=False)
ex1.download_button("⬇️ Export jadwal MINGGUAN (Excel)", out_w.getvalue(),
    file_name=f"jadwal_mingguan_{week_lo.date()}_{week_hi.date()}.xlsx", mime=XLSX_MIME)

# --- export harian (hari terpilih) ---
if view_day is not None:
    out_d = BytesIO()
    with pd.ExcelWriter(out_d, engine="openpyxl") as w:
        disp[exp_cols].rename(columns=ren).sort_values(["grup", "urgency"]).to_excel(
            w, sheet_name="Jadwal_Harian", index=False)
        unit_summary(disp, speed).to_excel(w, sheet_name="Ringkasan_Unit", index=False)
    ex2.download_button(f"⬇️ Export jadwal HARIAN {view_day.date()} (Excel)", out_d.getvalue(),
        file_name=f"jadwal_harian_{view_day.date()}.xlsx", mime=XLSX_MIME, type="primary")
else:
    ex2.caption("Pilih satu hari di toggle atas buat export jadwal harian.")
