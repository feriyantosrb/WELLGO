"""
WELLGO (Well Grouping Optimizer) — 9 Unit MWT Daily Planner
----------------------------------------------------------
Optimasi rute logistik well testing Sumatra Light North (SL North).
Terintegrasi dengan design system `wellgo_ui`.

Run: py -m streamlit run app.py
"""

import re
import os
import json
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
import plotly.express as px

st.set_page_config(page_title="WELLGO", page_icon="wellgo_icon.png", layout="wide")

ui.inject_theme()

DB_PATH = "welltest_status.db"
ROAD_PARQUET = "road_dist_cache.parquet"   # basis cache jarak jalan bawaan repo (bertahan lintas redeploy Streamlit Cloud)

def db_connect():
    """Koneksi SQLite dengan busy_timeout panjang + WAL. Di Streamlit Cloud satu file DB
    dipakai banyak sesi/rerun sekaligus; saat build matriks jalan menahan lock tulis, rerun
    lain yang memanggil init_db() bisa kena 'database is locked' setelah 5 dtk (default) lalu
    OperationalError. timeout=60 + busy_timeout membuat operasi MENUNGGU writer selesai, dan
    WAL mengizinkan baca sembari tulis sehingga tabrakan lock jauh berkurang."""
    con = sqlite3.connect(DB_PATH, timeout=60)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=60000")
        con.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    return con
SHEET_DEFAULT = "Compiled Schedule"
SHEET_UNITMAP = "Balam_South"   # sheet opsional: alokasi unit per (sub-area, field) — sumber kebenaran mutlak, bypass zona
SHEET_STATUS = "Status_Sumur"   # sheet opsional: overlay last_status & last_unit_name per (Rentang Periode, well_name)
SHEET_MAPUNIT = "Mapping_Unit"  # sheet opsional: daftar unit yang BOLEH menggarap tiap (SUB_AREA, FIELD) — dipakai mode Mapping Unit

REMOTE_AREAS = {"BANGKO", "BALAM"}
REMOTE_UNITS = ["MPAS_444", "MPAS_768", "MPAS_523", "MPAS_445", "MPAS_534"]
NONREMOTE_UNITS = ["MPAS_535", "MPAS_524", "MPAS_525", "MPAS_767"]
ALL_UNITS = REMOTE_UNITS + NONREMOTE_UNITS
ADDMAN_URG = 1_000_000   # urgensi sentinel utk Add Manual → paling bawah (isi slot sisa, jangan geser yang lain)

# ------------------------------------------------------------------ persistence
def init_db():
    con = db_connect()
    con.execute("""CREATE TABLE IF NOT EXISTS execution_log(
        plan_date TEXT, well_name TEXT, unit TEXT, status TEXT, reason TEXT, updated_at TEXT,
        PRIMARY KEY(plan_date, well_name))""")
    existing = {r[1] for r in con.execute("PRAGMA table_info(execution_log)").fetchall()}
    for col in ("unit", "status", "reason", "updated_at", "comment"):
        if col not in existing:
            con.execute(f"ALTER TABLE execution_log ADD COLUMN {col} TEXT")
    con.execute("""CREATE TABLE IF NOT EXISTS coord_cache(
        well_name TEXT PRIMARY KEY, lat REAL, lon REAL, updated_at TEXT)""")
    # Cache jarak jalan nyata (OSRM) per pasangan koordinat berarah, km.
    con.execute("""CREATE TABLE IF NOT EXISTS road_dist_cache(
        alat REAL, alon REAL, blat REAL, blon REAL, km REAL, updated_at TEXT,
        PRIMARY KEY(alat, alon, blat, blon))""")
    # Geometri polyline jalan nyata (OSRM /route) per pasangan koordinat berarah.
    con.execute("""CREATE TABLE IF NOT EXISTS road_geom_cache(
        alat REAL, alon REAL, blat REAL, blon REAL, geom TEXT, updated_at TEXT,
        PRIMARY KEY(alat, alon, blat, blon))""")
    con.commit()
    con.close()

def save_status(plan_date, rows):
    con = db_connect()
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
    con = db_connect()
    con.execute("DELETE FROM execution_log")
    con.commit()
    con.close()

def status_in_period(lo, hi):
    con = db_connect()
    try:
        q = ("SELECT well_name AS well, status, reason, comment, plan_date FROM execution_log "
             "WHERE status IN ('executed','ncmp','pending') AND plan_date BETWEEN ? AND ?")
        df = pd.read_sql(q, con, params=(str(lo), str(hi)))
    except Exception:
        df = pd.DataFrame(columns=["well", "status", "reason", "comment", "plan_date"])
    con.close()
    _empty_pend = pd.DataFrame(columns=["well", "plan_date"])
    if not len(df):
        return set(), pd.DataFrame(columns=["well", "reason", "comment", "plan_date"]), _empty_pend
    df["plan_date"] = df["plan_date"].astype(str)
    df["comment"] = df["comment"].fillna("").astype(str) if "comment" in df.columns else ""
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
            .sort_values("plan_date").groupby("well", as_index=False).last()[["well", "reason", "comment", "plan_date"]])
    pending = (latest[latest["well"].isin(pend_w)]
               .sort_values("plan_date").groupby("well", as_index=False).last()[["well", "plan_date"]])
    return executed, ncmp, pending

def comp_records(wells):
    """COMP (executed) per well dari execution_log → {well: [(Timestamp, reason_upper), ...]} (seluruh log).
    Dipakai utk deteksi fase AWS: kolom Reason (AS1/AS2) sbg sinyal utama, window POP sbg fallback."""
    con = db_connect()
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
    con = db_connect()
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

def xl_sheet(writer, df, sheet, name=None):
    """Tulis df ke sheet SEBAGAI Excel Table (ListObject), bukan range biasa:
    langsung ada autofilter, baris belang, dan header beku. Nama tabel dibersihkan
    (Excel menolak spasi/simbol & nama > 30 char)."""
    df.to_excel(writer, sheet_name=sheet, index=False)
    try:
        from openpyxl.worksheet.table import Table, TableStyleInfo
        from openpyxl.utils import get_column_letter
        ws = writer.sheets[sheet]
        nr, nc = len(df), len(df.columns)
        if nc == 0:
            return
        for i, col in enumerate(df.columns, 1):
            vals = df[col].astype(str).head(200).tolist() if nr else []
            wdt = max([len(str(col))] + [len(v) for v in vals]) + 2
            ws.column_dimensions[get_column_letter(i)].width = min(max(wdt, 10), 42)
        ws.freeze_panes = "A2"
        if nr:  # Excel Table butuh minimal 1 baris data
            nm = re.sub(r"\W", "_", str(name or sheet))[:30].strip("_") or "Tabel"
            t = Table(displayName=nm, ref=f"A1:{get_column_letter(nc)}{nr + 1}")
            t.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
            ws.add_table(t)
    except Exception:
        pass  # tabel gagal dibuat → sheet tetap terisi sbg range biasa

def kat_label(cat, tipe="", rtag=""):
    """Label kategori LENGKAP utk tooltip/tabel: REG A/B/C/D, AWS1/AWS2, NW1/NW2/NW3,
    ADD A/B/C/D — bukan sekadar REG/AWS/NW. Tag permintaan (PRQ/ORQ) ditempel di
    belakang karena sumber & maknanya beda (dari kolom Remark, bukan test_category)."""
    c = " ".join(str(cat).upper().replace("-", " ").replace("_", " ").split())
    m = re.search(r"NEW WELL\s*([123])", c)
    if m:
        lab = f"NW{m.group(1)}"
    elif "NEW WELL" in c:
        lab = "NW"
    elif "AWS" in c:
        m = re.search(r"AWS\s*([12])", c);  lab = f"AWS{m.group(1)}" if m else "AWS"
    elif "MANUAL" in c:
        m = re.search(r"\b([A-D])\b", c);   lab = f"ADD {m.group(1)}" if m else "ADD MAN"
    elif "REGULAR" in c or "RTN" in c:
        m = re.search(r"REGULAR\s*([A-D])\b", c); lab = f"REG {m.group(1)}" if m else "REG"
    else:
        lab = c[:14] if c and c != "NAN" else (str(tipe) or "-")
    rt = str(rtag).upper().strip()
    return f"{lab} · {rt}" if rt in ("PRQ", "ORQ") else lab

def classify_status(stat):
    s = str(stat).strip().upper().replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    if s.startswith("NCMP") or s.startswith("NOT COMP") or s.startswith("INCOMP"): return "NCMP"
    if s.startswith("COMP") or s in ("DONE", "OK", "C", "EXECUTED", "TESTED"): return "COMP"
    return ""

# ── COMMENT IF NOT COMPLETE → hambatan lapangan yang wajib dibawa ulang ────
# Kolom COMMENT IF NOT COMPLETE di SCH_Database menjelaskan MENGAPA tes gagal.
# Tiga kode di bawah berarti kegagalan bukan soal kapasitas kru: fasilitas belum
# siap (FACI), akses jalan tak bisa dilewati (ROAD), atau sumurnya mati saat kru
# tiba (WOFF). Sumur begini harus dijadwalkan ulang SEPANJANG sisa periode walau
# deadline (max_date) sudah terlewat, dan bila sampai akhir periode tetap tak
# kebagian slot, ia BUKAN Miss Deadline melainkan kategori tersendiri.
NCMP_CARRY_CODES = ("FACI", "ROAD", "WOFF")
NCMP_CARRY_LABEL = "Not Complete (NCMP) FACI/ROAD/WOFF-Miss Deadline"

def carry_code(comment):
    """Kode hambatan (FACI/ROAD/WOFF) dari teks COMMENT IF NOT COMPLETE; '' bila bukan.
    Cocok di awal kata sehingga 'FACILITY NOT READY' & 'ROAD ACCESS' ikut terbaca."""
    s = str(comment).upper()
    return next((c for c in NCMP_CARRY_CODES if re.search(rf"\b{c}", s)), "")

def carry_label(code):
    return f"Not Complete (NCMP) {code}-Miss Deadline" if code else NCMP_CARRY_LABEL

SCHDB_COLS = ["well", "unit", "date", "raw_stat", "stat", "reason", "comment"]

def read_schdb(file_bytes):
    """Baca satu file SCHDatabase jadi tabel ternormalisasi. MURNI parsing — tak menyentuh
    execution_log — supaya bisa dipakai dua jalur: impor status harian (yang memang menulis
    ke database) dan tarikan riwayat untuk komparasi (yang tidak boleh menulis apa pun)."""
    try:
        xls = pd.ExcelFile(BytesIO(file_bytes))
    except Exception:
        return pd.DataFrame(columns=SCHDB_COLS)
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
    # REASON = alasan tes (AS1/AS2 dst, dipakai deteksi fase AWS).
    # COMMENT IF NOT COMPLETE = hambatan saat gagal (FACI/ROAD/WOFF) — kolom berbeda,
    # jadi disimpan terpisah. File lama tanpa REASON tetap jatuh ke kolom comment.
    cc = cols.get("COMMENT IF NOT COMPLETE")
    cr = cols.get("REASON") or cc
    if not (cw and cs and cd):
        return pd.DataFrame(columns=SCHDB_COLS)
    w = pd.DataFrame({
        "well": df[cw].astype(str).str.strip(),
        "raw_stat": df[cs].astype(str).str.strip().str.upper(),
        "date": pd.to_datetime(df[cd], errors="coerce"),
        "unit": df[cu].map(norm_unit) if cu else "",
        "reason": (df[cr].astype(str).str.strip().str.upper().replace({"NAN": ""}) if cr else ""),
        "comment": (df[cc].astype(str).str.strip().str.upper().replace({"NAN": ""}) if cc else ""),
    })
    w["stat"] = w["raw_stat"].map(classify_status)
    return w

def sch_history(file_list):
    """Riwayat jadwal tes manual dari file SCHDatabase yang diunggah KHUSUS untuk komparasi.
    Sengaja tidak lewat import_compncmp: file ini hanya menjawab "siapa dijadwalkan ke unit
    mana pada tanggal berapa", dan TIDAK BOLEH ikut menentukan COMP/NCMP/PENDING maupun
    kelayakan penjadwalan. Hanya unit MPAS yang dipakai — unit TS tak punya padanan di WELLGO."""
    frames = [w for w in (read_schdb(fb) for fb in file_list) if len(w)]
    if not frames:
        return pd.DataFrame(columns=["well", "unit", "date", "stat", "reason"])
    h = pd.concat(frames, ignore_index=True)
    h = h[~h["well"].isin(["", "nan"]) & h["date"].notna()]
    h = h[h["unit"].astype(str).str.startswith("MPAS")]
    return h.drop_duplicates(["well", "date", "unit"])[["well", "unit", "date", "stat", "reason"]].copy()

def import_compncmp(file_list):
    n_comp = n_ncmp = n_pend = 0
    reasons = {}
    status_seen = {}
    skip_date = skip_well = skip_status = 0
    con = db_connect()
    now = datetime.now().isoformat(timespec="seconds")
    for fb in file_list:
        w = read_schdb(fb)
        if not len(w): continue
        for k, v in w["raw_stat"].value_counts().items():
            status_seen[k] = status_seen.get(k, 0) + int(v)
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
                        w["comment"], [now] * len(w)))
        con.executemany("""INSERT INTO execution_log(plan_date,well_name,unit,status,reason,comment,updated_at)
            VALUES(?,?,?,?,?,?,?) ON CONFLICT(plan_date,well_name) DO UPDATE SET
            unit=excluded.unit, status=excluded.status, reason=excluded.reason,
            comment=excluded.comment, updated_at=excluded.updated_at""", rows)
        n_comp += int((w["stat"] == "COMP").sum())
        nc = w[w["stat"] == "NCMP"]
        n_ncmp += len(nc)
        _why = nc["comment"].where(nc["comment"].astype(bool), nc["reason"])
        for rsn, cnt in _why.replace("", "(kosong)").value_counts().items():
            reasons[rsn] = reasons.get(rsn, 0) + int(cnt)
    con.commit()
    con.close()
    return {"comp": n_comp, "ncmp": n_ncmp, "pending": n_pend, "reasons": reasons, "status_seen": status_seen,
            "skip_date": skip_date, "skip_well": skip_well, "skip_status": skip_status}

def save_coords(pairs):
    con = db_connect()
    now = datetime.now().isoformat(timespec="seconds")
    for well, lat, lon in pairs:
        if pd.notna(lat) and pd.notna(lon):
            con.execute("""INSERT INTO coord_cache(well_name,lat,lon,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(well_name) DO UPDATE SET lat=excluded.lat, lon=excluded.lon,
                updated_at=excluded.updated_at""", (well, float(lat), float(lon), now))
    con.commit()
    con.close()

def load_coord_cache():
    con = db_connect()
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

# ── Mode Mapping Unit: daftar unit yang boleh menggarap tiap (SUB_AREA, FIELD) ──
# Sheet Mapping_Unit hidup di file Excel yang sama dengan kandidat sumur:
#   SUB_AREA | FIELD        | UNIT
#   BALAMN   | ANTARA       | MP445, MP523
#   BALAMN   | MENGGALA_NO  | MP768
# Beda dengan forced_unit yang mengunci 1 sumur ke 1 unit, di sini satu lapangan
# boleh punya beberapa unit dan sumurnya tetap berkompetisi seperti biasa —
# yang dibatasi cuma unit MANA saja yang berhak mengambilnya.
@st.cache_data(show_spinner=False)
def load_unit_map(file_bytes, sheet=SHEET_MAPUNIT):
    """Return {(sub_area, field): (unit, ...)}. Sub-area kosong = berlaku utk semua sub-area."""
    try:
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    except Exception:
        return {}
    df.columns = [str(c).strip().upper() for c in df.columns]
    cf = "FIELD" if "FIELD" in df.columns else None
    cu = next((c for c in ("UNIT", "UNITS") if c in df.columns), None)
    cs = next((c for c in ("SUB_AREA", "SUBAREA", "OP_SUB_AREA_CODE") if c in df.columns), None)
    if not (cf and cu):
        return {}
    umap = {}
    for _, r in df.iterrows():
        f = str(r[cf]).upper().strip()
        if not f or f == "NAN":
            continue
        # satu sel boleh memuat banyak unit: "MP445, MP523" → (MPAS_445, MPAS_523)
        units = tuple(dict.fromkeys(norm_unit(u) for u in re.split(r"[,;/|]+", str(r[cu]))
                                    if u.strip() and u.strip().upper() != "NAN"))
        if not units:
            continue
        sa = str(r[cs]).upper().strip() if cs else ""
        umap[("" if sa == "NAN" else sa, f)] = units
    return umap

def unit_map_allow(df, umap):
    by_field = {}
    for (_sa, f), units in umap.items():
        cur = by_field.get(f, ())
        by_field[f] = cur + tuple(u for u in units if u not in cur)
    fld = df["field"].astype(str).str.upper().str.strip()
    return [by_field.get(f) for f in fld]

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
    lw_col = next((c for c in df.columns if "LAST" in str(c).upper()
                   and ("WT" in str(c).upper() or "WELL TEST" in str(c).upper())), None)
    df["last_wt"] = to_dt(df[lw_col]) if lw_col else pd.NaT

    df["status_src"] = "compiled"
    try:
        so = pd.read_excel(BytesIO(file_bytes), sheet_name=SHEET_STATUS)
        so.columns = [str(c).strip() for c in so.columns]
        _pick = lambda cols, names: next((c for c in cols if str(c).strip().lower() in names), None)
        wcol = _pick(so.columns, {"well_name", "well"})
        scol = _pick(so.columns, {"last_status", "well status", "status"})
        ucol = _pick(so.columns, {"last_unit_name", "unit_name", "unit"})
        tcol = _pick(so.columns, {"string_type", "string type"})
        pcol = next((c for c in so.columns if "RENTANG" in str(c).upper()), None)
        dpcol = next((c for c in df.columns if "RENTANG" in str(c).upper()), None)
        if wcol and (scol or ucol or tcol):
            _k = lambda s: (s.astype(str).str.strip().str.upper()
                            .str.replace(r"\s+", " ", regex=True))
            so = so[[c for c in (pcol, wcol, scol, ucol, tcol) if c]].copy()
            so["_kw"] = _k(so[wcol])
            df["_kw"] = _k(df["well"])
            keys = ["_kw"]
            if pcol and dpcol:                     
                so["_kp"] = _k(so[pcol]); df["_kp"] = _k(df[dpcol]); keys = ["_kp", "_kw"]
            so = so.drop_duplicates(subset=keys, keep="last")
            m = df[keys].merge(so, on=keys, how="left")
            m.index = df.index
            _has = lambda s: s.notna() & (s.astype(str).str.strip().str.upper() != "NAN") \
                             & (s.astype(str).str.strip() != "")
            hit = pd.Series(False, index=df.index)
            def _apply(col, sc):
                nonlocal hit
                if col not in df.columns:
                    df[col] = pd.Series(pd.NA, index=df.index, dtype="object")
                elif df[col].dtype != object:
                    df[col] = df[col].astype(object)
                ok = _has(m[sc]); df.loc[ok, col] = m.loc[ok, sc]; hit = hit | ok
            if ucol: _apply("unit", ucol)
            if scol: _apply("last_status", scol)
            if tcol: _apply("string_type", tcol)
            df.loc[hit, "status_src"] = SHEET_STATUS
            df = df.drop(columns=[c for c in ("_kw", "_kp") if c in df.columns])
    except Exception:
        pass 

    df["unit"] = df["unit"].map(norm_unit_name)

    st_ = df["string_type"].astype(str).str.upper().str.strip()
    area_ = df["area"].astype(str).str.upper().str.strip()
    fld_ = df["field"].astype(str).str.upper().str.strip()
    
    df["forced_unit"] = None
    # Aturan Mutlak GP tetap dipertahankan
    df.loc[st_.eq("GP") & area_.eq("BEKASAP"), "forced_unit"] = "MPAS_525"
    df.loc[st_.eq("GP") & area_.isin(["BANGKO", "BALAM"]), "forced_unit"] = "MPAS_768"
    # REVISI: Aturan mutlak "BENAR -> 534" DIHAPUS. BENAR akan ikut kompetisi deadline di Remote Area.

    # Override alokasi Balam_South (Revisi X-Ray)
    try:
        _bs = pd.read_excel(BytesIO(file_bytes), sheet_name=SHEET_UNITMAP)
        _bs.columns = [str(c).strip().upper() for c in _bs.columns]
        if {"FIELD", "UNIT"}.issubset(_bs.columns):
            _has_sub = "OP_SUB_AREA_CODE" in _bs.columns
            _umap = {}
            for _, _r in _bs.iterrows():
                _f = str(_r["FIELD"]).upper().strip()
                _u = norm_unit(_r["UNIT"])
                if not _f or _f == "NAN" or not _u or str(_u).upper() == "NAN":
                    continue
                _s = str(_r["OP_SUB_AREA_CODE"]).upper().strip() if _has_sub else ""
                _umap[(_s, _f)] = _u
            sub_ = df["subarea"].astype(str).str.upper().str.strip()
            _keys = zip((sub_ if _has_sub else pd.Series([""] * len(df), index=df.index)), fld_)
            _mapped = pd.Series([_umap.get(k) for k in _keys], index=df.index)
            
            # REVISI ATURAN BALAM SOUTH:
            # 1. MPAS_767: Balam yg tadinya buat 767 ditarik paksa ke zona Non-Remote ("BEKASAP") agar 
            #    berkompetisi memperebutkan seluruh unit Non-Remote berdasarkan deadline.
            df.loc[_mapped == "MPAS_767", "area"] = "BEKASAP"
            
            # 2. MPAS_525: Tetap terkunci ke MPAS_525.
            df.loc[_mapped == "MPAS_525", "forced_unit"] = "MPAS_525"
            
            # 3. MPAS_768: Tidak dikunci. Karena area aslinya Remote, dia akan otomatis berkompetisi 
            #    memperebutkan seluruh unit Remote berdasarkan deadline. (Tidak perlu ada kode).
            
    except Exception:
        pass

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
    df["is_addmanual"] = cat_u.str.contains("MANUAL", na=False) & (df["req_tag"] == "") & ~cat_force
    df["is_gp"] = st_.eq("GP")
    df["is_reg_a"] = cat_u.str.contains(r"REGULAR\s*A\b", regex=True, na=False)
    df["kat_full"] = [kat_label(c, t, r) for c, t, r in zip(df["category"], df["tipe"], df["req_tag"])]

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
    D = _dist_matrix(lat, lon)
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
    key = (bool(_USE_ROAD), tuple(sorted(zip(np.round(lat, 5).tolist(), np.round(lon, 5).tolist()))))
    store = _route_cache_store()
    val = store.get(key)
    if val is None:
        val = _solve_route(lat, lon)[1]
        if len(store) > 50000: store.clear()
        store[key] = val
    return val

def optimize_route(lat, lon):
    return _solve_route(lat, lon)

# ── Jarak jalan nyata (OSRM /table) ─────────────────────────────────────────
# Semua optimasi & KPI km mengalir lewat _dist_matrix(). Default = haversine
# (garis lurus). Bila _USE_ROAD True & pasangan sumur ada di cache jalan (_ROAD_KM),
# jaraknya dipakai; pasangan yang belum ada JATUH ke haversine, jadi tak pernah patah.
_ROAD_KM = {}            # {((rlat,rlon),(rlat,rlon)): km}  berarah
_USE_ROAD = False        # di-set dari sidebar setelah cache dimuat
_ROAD_DETOUR = 1.0       # rasio khas km_jalan/km_lurus dari cache, utk skala pasangan tak tercache
_OSRM_INSECURE = False   # abaikan verifikasi sertifikat SSL panggilan OSRM (proxy korporat)

def _osrm_ctx():
    """SSLContext tanpa verifikasi bila _OSRM_INSECURE aktif (untuk proxy korporat yang
    menyisipkan sertifikat sendiri), selain itu None = verifikasi normal."""
    if not _OSRM_INSECURE:
        return None
    import ssl
    return ssl._create_unverified_context()

def _rk(lat, lon):
    return (round(float(lat), 5), round(float(lon), 5))

def road_detour_factor(cache):
    """Rasio khas (median) km jalan terhadap km garis lurus dari isi cache. Dipakai
    menskala estimasi pasangan yang BELUM tercache supaya satu satuan dgn km jalan,
    jadi pengelompokan tak terdistorsi saat cache belum lengkap."""
    rs = []
    for (a, b), km in cache.items():
        h = haversine_km(a[0], a[1], b[0], b[1])
        if h > 0.05 and km > 0:
            rs.append(km / h)
    if not rs:
        return 1.0
    f = float(np.median(rs))
    return f if f >= 1.0 else 1.0

def _pack_coords(lat, lon):
    """Kemas (lat,lon) bulat-5-desimal jadi satu int64 untuk penyaringan vektor cepat."""
    ilat = np.rint(np.asarray(lat, dtype=float) * 1e5).astype(np.int64)
    ilon = np.rint(np.asarray(lon, dtype=float) * 1e5).astype(np.int64)
    return ilat * np.int64(20_000_000) + ilon

def _dict_from_dist_df(df, need_arr=None):
    """Dict {((alat,alon),(blat,blon)): km} dari DataFrame. Bila need_arr diberikan (array
    int64 hasil _pack_coords), HANYA pasangan yang KEDUA ujungnya ada di need_arr yang diambil,
    supaya dict tetap kecil di memori (cegah OOM di Streamlit Cloud saat cache jutaan pasangan)."""
    if df is None or not len(df):
        return {}
    if need_arr is not None:
        ka = _pack_coords(df["alat"].values, df["alon"].values)
        kb = _pack_coords(df["blat"].values, df["blon"].values)
        mask = np.isin(ka, need_arr) & np.isin(kb, need_arr)
        df = df[mask]
        if not len(df):
            return {}
    a = zip(df["alat"].tolist(), df["alon"].tolist())
    b = zip(df["blat"].tolist(), df["blon"].tolist())
    return dict(zip(zip(a, b), df["km"].astype(float).tolist()))

def _need_arr(need):
    """need: iterable koordinat (lat,lon) sumur relevan → array int64 unik untuk penyaringan.
    None/kosong → None (muat semua; hati-hati untuk cache besar)."""
    if not need:
        return None
    pts = np.array(list(need), dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return None
    return np.unique(_pack_coords(pts[:, 0], pts[:, 1]))

def _parquet_pairs(need):
    """Baca HANYA baris relevan dari Parquet lewat predicate pushdown pyarrow: filter kolom
    koordinat langsung di file (lat/lon sisi-A & sisi-B harus termasuk sumur `need`), sehingga
    yang masuk ke memori hanya ribuan baris terkait, bukan jutaan baris seluruh matriks.
    Inilah yang membuat run ringan di Streamlit Cloud: tak ada lagi baca+kemas 4,8 juta baris
    tiap ganti periode, dan tak ada array raksasa yang ditahan di memori sepanjang sesi.
    Penyaringan tepat pasangan berarah tetap dilakukan pemanggil lewat _dict_from_dist_df."""
    empty = pd.DataFrame(columns=["alat", "alon", "blat", "blon", "km"])
    if not need or not os.path.exists(ROAD_PARQUET):
        return empty
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        pts = np.array(list(need), dtype=float)
        # koordinat di Parquet sudah dibulatkan 5 desimal; samakan agar cocok persis.
        lats = pa.array(np.unique(np.rint(pts[:, 0] * 1e5) / 1e5))
        lons = pa.array(np.unique(np.rint(pts[:, 1] * 1e5) / 1e5))
        flt = (ds.field("alat").isin(lats) & ds.field("alon").isin(lons) &
               ds.field("blat").isin(lats) & ds.field("blon").isin(lons))
        tbl = ds.dataset(ROAD_PARQUET, format="parquet").to_table(
            columns=["alat", "alon", "blat", "blon", "km"], filter=flt)
        return tbl.to_pandas()
    except Exception:
        # Fallback aman bila pyarrow.dataset tak tersedia: baca kolom lalu biar pemanggil saring.
        try:
            return pd.read_parquet(ROAD_PARQUET, columns=["alat", "alon", "blat", "blon", "km"])
        except Exception:
            return empty

def load_road_dist(need=None):
    # Basis Parquet (dibaca predicate pushdown → hanya baris terkait) + tambahan SQLite,
    # DISARING ke pasangan antar sumur `need`. Tak ada lagi scan/penyimpanan jutaan baris.
    na = _need_arr(need)
    d = {}
    # Hanya ambil dari Parquet bila ada daftar sumur (na). na None = tak menyaring → jangan
    # bangun dict penuh jutaan pasangan (cegah OOM); biarkan hanya tambahan SQLite yang dipakai.
    if na is not None:
        d = _dict_from_dist_df(_parquet_pairs(need), na)
    con = db_connect()
    try:
        sdf = pd.read_sql("SELECT alat,alon,blat,blon,km FROM road_dist_cache", con)
    except Exception:
        sdf = pd.DataFrame(columns=["alat", "alon", "blat", "blon", "km"])
    con.close()
    if len(sdf):
        d.update(_dict_from_dist_df(sdf, na))
    return d

def _road_dist_count():
    """Jumlah total pasangan tersimpan (Parquet + SQLite) untuk ditampilkan — TANPA memuat
    dict-nya. Baca num_rows dari metadata Parquet (murah) + COUNT SQLite."""
    n = 0
    try:
        import pyarrow.parquet as _pq
        n += int(_pq.ParquetFile(ROAD_PARQUET).metadata.num_rows)
    except Exception:
        pass
    con = db_connect()
    try:
        n += int(con.execute("SELECT COUNT(*) FROM road_dist_cache").fetchone()[0])
    except Exception:
        pass
    finally:
        con.close()
    return n

def _road_dist_sig():
    """Sidik cepat sumber cache (Parquet repo + SQLite) sebagai kunci cache. Berubah bila
    Parquet berganti (redeploy) atau SQLite bertambah, sehingga dict otomatis dimuat ulang."""
    con = db_connect()
    try:
        row = con.execute("SELECT COUNT(*), COALESCE(MAX(updated_at),'') FROM road_dist_cache").fetchone()
        base = (int(row[0]), str(row[1]))
    except Exception:
        base = (0, "")
    finally:
        con.close()
    try:
        _st = os.stat(ROAD_PARQUET)
        pq = (int(_st.st_size), int(_st.st_mtime))
    except Exception:
        pq = (0, 0)
    return base + pq

@st.cache_resource(show_spinner=False)
def load_road_dist_cached(sig, need_hash, _need=None):
    """Muat cache jarak jalan (DISARING ke sumur `_need`) + faktor detour, SEKALI per
    (isi cache, set sumur). Kunci cache = (sig, need_hash) yang KECIL, sedangkan `_need`
    (berawalan _) TIDAK ikut di-hash Streamlit. Ini penting: meng-hash frozenset ribuan
    koordinat tiap rerun lambat, jadi kita hash sendiri sekali dan berikan int-nya saja."""
    d = load_road_dist(_need)
    return d, road_detour_factor(d)

def save_road_dist(pairs):
    """pairs: iterable of ((rlat,rlon),(rlat,rlon), km)."""
    con = db_connect()
    now = datetime.now().isoformat(timespec="seconds")
    con.executemany("""INSERT INTO road_dist_cache(alat,alon,blat,blon,km,updated_at)
        VALUES(?,?,?,?,?,?) ON CONFLICT(alat,alon,blat,blon) DO UPDATE SET
        km=excluded.km, updated_at=excluded.updated_at""",
        [(a[0], a[1], b[0], b[1], float(km), now) for a, b, km in pairs])
    con.commit()
    con.close()

def osrm_build_matrix(coords, base_url, chunk=40, timeout=120, retries=3, profile="driving", progress=None):
    """Bangun matriks jarak jalan berarah untuk daftar koordinat unik (lat,lon) via OSRM
    /table. Return (n_pair, n_null). Hasil disimpan ke road_dist_cache. Query di-blok
    agar aman untuk server dgn batas jumlah titik (mis. demo publik ~100). Tiap blok
    di-retry dgn backoff saat timeout/gangguan jaringan, lalu hasil parsial tetap disimpan
    supaya progres tak hilang bila server publik lambat."""
    import json, time, urllib.request, urllib.parse, urllib.error
    base = base_url.rstrip("/")
    pts = list(dict.fromkeys(_rk(la, lo) for la, lo in coords))   # unik, urut stabil
    n = len(pts)
    out, n_null = [], 0

    def _one(src_idx, dst_idx):
        idx = sorted(set(src_idx) | set(dst_idx))
        remap = {gi: k for k, gi in enumerate(idx)}
        coord_str = ";".join(f"{pts[gi][1]:.6f},{pts[gi][0]:.6f}" for gi in idx)  # lon,lat
        q = urllib.parse.urlencode({
            "annotations": "distance",
            "sources": ";".join(str(remap[gi]) for gi in src_idx),
            "destinations": ";".join(str(remap[gi]) for gi in dst_idx)})
        url = f"{base}/table/v1/{profile}/{coord_str}?{q}"
        last = None
        for att in range(max(1, retries)):
            try:
                with urllib.request.urlopen(url, timeout=timeout, context=_osrm_ctx()) as resp:
                    data = json.loads(resp.read().decode())
                if data.get("code") != "Ok":
                    raise RuntimeError(f"OSRM: {data.get('code')} {data.get('message', '')}")
                return data["distances"]      # meter; bisa null bila tak terjangkau
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                if att < retries - 1:
                    time.sleep(2 ** att)      # backoff 1s, 2s, 4s…
        raise RuntimeError(f"OSRM tak merespons setelah {retries} percobaan: {last}. "
                           "Coba kecilkan chunk, perbesar timeout, atau pakai server OSRM sendiri.")

    blocks = [list(range(i, min(i + chunk, n))) for i in range(0, n, chunk)]
    total = len(blocks) * len(blocks)
    done = saved = 0
    try:
        for bs in blocks:
            for bd in blocks:
                dm = _one(bs, bd)
                for si, gi in enumerate(bs):
                    for di, gj in enumerate(bd):
                        if gi == gj:
                            continue
                        m = dm[si][di]
                        if m is None:
                            n_null += 1
                            continue
                        out.append((pts[gi], pts[gj], float(m) / 1000.0))
                done += 1
                if progress:
                    progress(done / total)
                # Flush berkala: tiap tulisan pendek (lock singkat), memori & WAL terjaga,
                # dan progres tersimpan bertahap alih-alih satu tulisan raksasa di akhir.
                if len(out) >= 20000:
                    save_road_dist(out)
                    saved += len(out)
                    out = []
    finally:
        if out:                                # simpan sisa / hasil parsial walau ada gangguan
            save_road_dist(out)
            saved += len(out)
    return saved, n_null

def _dist_matrix(lat, lon):
    """Matriks jarak N×N untuk optimasi. Jalan nyata bila aktif & tersedia, sisanya haversine."""
    H = _haversine_matrix(lat, lon)
    if not _USE_ROAD or not _ROAD_KM:
        return H
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    keys = [_rk(lat[i], lon[i]) for i in range(len(lat))]
    # Pasangan tak tercache diperkirakan = km lurus × detour khas, supaya satu satuan
    # dengan km jalan yang tercache (hindari campur metrik saat cache belum lengkap).
    D = H * _ROAD_DETOUR if _ROAD_DETOUR and _ROAD_DETOUR > 1.0 else H.copy()
    np.fill_diagonal(D, 0.0)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j:
                continue
            v = _ROAD_KM.get((keys[i], keys[j]))
            if v is not None:
                D[i, j] = v
    return D

# ── Geometri rute jalan nyata (OSRM /route) untuk gambar di peta ─────────────
_ROAD_GEOM = {}          # {((alat,alon),(blat,blon)): [[lon,lat],…]} berarah a→b
_DRAW_ROAD = False       # di-set dari sidebar
_GEOM_FAIL = False       # short-circuit sesi bila OSRM tak terjangkau saat render
_GEOM_FETCH_N = 0        # jumlah geometri diambil live pada render ini (dibatasi agar render tetap responsif)
_GEOM_FETCH_MAX = 40     # plafon fetch OSRM /route per render; sisanya garis lurus & ter-cache bertahap
_GEOM_TO_CAP = 8         # batas atas timeout per pasangan saat render (detik) — cegah render menggantung

def load_road_geom():
    con = db_connect()
    try:
        df = pd.read_sql("SELECT alat,alon,blat,blon,geom FROM road_geom_cache", con)
    except Exception:
        df = pd.DataFrame(columns=["alat", "alon", "blat", "blon", "geom"])
    con.close()
    out = {}
    for r in df.itertuples():
        try:
            out[((r.alat, r.alon), (r.blat, r.blon))] = json.loads(r.geom)
        except Exception:
            continue
    return out

def save_road_geom(items):
    """items: iterable of ((alat,alon),(blat,blon), geom_list)."""
    con = db_connect()
    now = datetime.now().isoformat(timespec="seconds")
    con.executemany("""INSERT INTO road_geom_cache(alat,alon,blat,blon,geom,updated_at)
        VALUES(?,?,?,?,?,?) ON CONFLICT(alat,alon,blat,blon) DO UPDATE SET
        geom=excluded.geom, updated_at=excluded.updated_at""",
        [(a[0], a[1], b[0], b[1], json.dumps(g), now) for a, b, g in items])
    con.commit()
    con.close()

def _osrm_route_geom(a, b, base_url, timeout=30, retries=2, profile="driving"):
    """a,b = kunci (lat,lon). Return list [[lon,lat],…] geometri jalan a→b, atau None."""
    import urllib.request, urllib.error, time
    base = base_url.rstrip("/")
    url = (f"{base}/route/v1/{profile}/{a[1]:.6f},{a[0]:.6f};{b[1]:.6f},{b[0]:.6f}"
           "?overview=full&geometries=geojson")
    for att in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_osrm_ctx()) as resp:
                data = json.loads(resp.read().decode())
            if data.get("code") != "Ok" or not data.get("routes"):
                return None
            return data["routes"][0]["geometry"]["coordinates"]   # [[lon,lat],…]
        except (urllib.error.URLError, TimeoutError, OSError):
            if att < retries - 1:
                time.sleep(1)
    raise RuntimeError("OSRM /route tak merespons")

def road_route_path(lons, lats, order, base_url, timeout=30, profile="driving"):
    """Bangun satu path [[lon,lat],…] menyusuri jalan nyata untuk urutan `order`.
    Pasangan tanpa geometri di-cache (fetch sekali) atau jatuh ke garis lurus.
    Bila OSRM gagal sekali, sisa render pakai garis lurus (short-circuit sesi)."""
    global _GEOM_FAIL, _GEOM_FETCH_N
    path, new = [], []
    _to = min(int(timeout) if timeout else _GEOM_TO_CAP, _GEOM_TO_CAP)
    def _push(seg):
        if path and path[-1] == seg[0]:
            path.extend(seg[1:])
        else:
            path.extend(seg)
    for t in range(len(order) - 1):
        i, j = order[t], order[t + 1]
        A, B = _rk(lats[i], lons[i]), _rk(lats[j], lons[j])
        g = _ROAD_GEOM.get((A, B))
        # Ambil geometri live hanya bila belum di-cache, OSRM belum gagal, dan plafon fetch
        # per render belum terlampaui. Timeout dibatasi (_GEOM_TO_CAP) + sekali percobaan agar
        # satu pasangan lambat/tak terjangkau tidak membuat render menggantung bermenit-menit.
        if g is None and not _GEOM_FAIL and _GEOM_FETCH_N < _GEOM_FETCH_MAX:
            _GEOM_FETCH_N += 1
            try:
                g = _osrm_route_geom(A, B, base_url, _to, retries=1, profile=profile)
            except Exception:
                _GEOM_FAIL = True
                g = None
            if g:
                _ROAD_GEOM[(A, B)] = g
                new.append((A, B, g))
        _push(g if g else [[lons[i], lats[i]], [lons[j], lats[j]]])
    if new:
        try:
            save_road_geom(new)
        except Exception:
            pass
    return path

# Palet warna per unit utk ekspor KML (RRGGBB) — cukup kontras utk 14 unit
_KML_PALETTE = ["E6194B", "3CB44B", "4363D8", "F58231", "911EB4", "42D4F4", "F032E6",
                "BFEF45", "FABED4", "469990", "9A6324", "800000", "808000", "000075"]

def _kml_color(hexc):
    # RRGGBB → format KML aabbggrr (alpha penuh)
    r, g, b = hexc[0:2], hexc[2:4], hexc[4:6]
    return f"ff{b}{g}{r}".lower()

def build_kml(df, title="WELLGO Route"):
    """KML titik sumur + garis rute per unit-hari, warna per unit.
    Bisa di-import ke Google My Maps (mymaps.google.com) atau Google Earth."""
    import html as _html
    d = df[df["lat"].notna() & df["lon"].notna()].copy()
    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{_html.escape(str(title))}</name>']
    units = sorted([str(u) for u in d["plan_unit"].dropna().unique()])
    uidx = {u: i for i, u in enumerate(units)}
    for i, u in enumerate(units):
        kc = _kml_color(_KML_PALETTE[i % len(_KML_PALETTE)])
        P.append(f'<Style id="u{i}"><IconStyle><color>{kc}</color><scale>1.1</scale>'
                 f'<Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-circle.png</href></Icon></IconStyle>'
                 f'<LineStyle><color>{kc}</color><width>4</width></LineStyle></Style>')
    for day_idx, dgrp in d.groupby("day_idx"):
        _pd = dgrp["plan_day"].iloc[0] if "plan_day" in dgrp.columns and len(dgrp) else None
        dstr = pd.Timestamp(_pd).strftime("%d %b %Y") if pd.notna(_pd) else "-"
        P.append(f'<Folder><name>Hari {int(day_idx)} — {dstr}</name>')
        for u, g in dgrp.groupby("plan_unit"):
            g = g.reset_index(drop=True)
            sid = f'u{uidx.get(str(u), 0)}'
            P.append(f'<Folder><name>{_html.escape(str(u))} ({len(g)} sumur)</name>')
            for _, r in g.iterrows():
                dl = pd.Timestamp(r["max_date"]).strftime("%Y-%m-%d") if pd.notna(r.get("max_date")) else "-"
                # Tag kategori ringkas: NW / AWS / PRQ / ORQ / RTN (sama seperti di peta app).
                _tp = str(r.get("tipe", "")).upper()
                _rt = str(r.get("req_tag", "")).upper()
                _cat = ("NW" if _tp == "NW" else "AWS" if _tp == "AWS"
                        else "PRQ" if _rt == "PRQ" else "ORQ" if _rt == "ORQ" else "RTN")
                _durv = r.get("dur")
                _dur = f"{int(round(float(_durv)))} menit" if pd.notna(_durv) else "-"
                # Nama placemark = label pin yang tampil di Google My Maps: Well · Kategori · Durasi.
                _pin = f'{r["well"]} · {_cat} · {_dur}'
                desc = (f"Kategori: {_cat} ({_html.escape(str(r.get('category','-')))}) | Durasi tes: {_dur}<br/>"
                        f"Unit: {_html.escape(str(u))} | Hari {int(day_idx)} ({dstr})<br/>"
                        f"Sub-area: {_html.escape(str(r.get('subarea','-')))} | Deadline: {dl}")
                P.append(f'<Placemark><name>{_html.escape(_pin)}</name>'
                         f'<styleUrl>#{sid}</styleUrl><description><![CDATA[{desc}]]></description>'
                         f'<Point><coordinates>{r["lon"]},{r["lat"]},0</coordinates></Point></Placemark>')
            if len(g) > 1:
                order, _ = optimize_route(g["lat"].values, g["lon"].values)
                if _DRAW_ROAD:
                    p = road_route_path(g["lon"].values, g["lat"].values, order,
                                        globals().get("_osrm_url_cfg", "https://router.project-osrm.org"),
                                        globals().get("_osrm_to_cfg", 30),
                                        globals().get("_osrm_profile_cfg", "driving"))
                    coords = " ".join(f'{lon},{lat},0' for lon, lat in p)
                else:
                    coords = " ".join(f'{g.loc[i,"lon"]},{g.loc[i,"lat"]},0' for i in order)
                P.append(f'<Placemark><name>Rute {_html.escape(str(u))} (Hari {int(day_idx)})</name>'
                         f'<styleUrl>#{sid}</styleUrl><LineString><tessellate>1</tessellate>'
                         f'<coordinates>{coords}</coordinates></LineString></Placemark>')
            P.append('</Folder>')
        P.append('</Folder>')
    P.append('</Document></kml>')
    return "\n".join(P)

def build_route_html(df, title="WELLGO Route"):
    """Peta rute mandiri (satu file HTML) memakai Leaflet + tile OpenStreetMap. Tiap titik
    diberi label permanen nomor urut + nama sumur, dan popup berisi kategori, durasi, unit,
    deadline. Garis rute per unit mengikuti urutan TSP. Dibuka langsung di browser tanpa
    upload apa pun dan tanpa akun Google (butuh internet untuk memuat peta dasar)."""
    import json as _json, html as _html
    d = df.copy()
    if "has_coord" in d.columns:
        d = d[d["has_coord"].fillna(False)]
    d = d[d["lat"].notna() & d["lon"].notna()].copy()
    palette = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
               "#bf9b30", "#469990", "#9A6324", "#800000", "#808000", "#000075", "#e6194B"]
    units = list(dict.fromkeys(str(u) for u in d["plan_unit"])) if "plan_unit" in d.columns else []
    ucol = {u: palette[i % len(palette)] for i, u in enumerate(units)}
    markers, routes = [], []
    _grp = d.groupby(["day_idx", "plan_unit"]) if "day_idx" in d.columns else d.groupby("plan_unit")
    for key, g in _grp:
        u = key[1] if isinstance(key, tuple) else key
        g = g.reset_index(drop=True)
        order, _ = optimize_route(g["lat"].values, g["lon"].values)
        col = ucol.get(str(u), "#4363d8")
        pts = []
        for seq, idx in enumerate(order, start=1):
            r = g.loc[idx]
            tp = str(r.get("tipe", "")).upper()
            rt = str(r.get("req_tag", "")).upper()
            cat = ("NW" if tp == "NW" else "AWS" if tp == "AWS"
                   else "PRQ" if rt == "PRQ" else "ORQ" if rt == "ORQ" else "RTN")
            durv = r.get("dur")
            dur = f"{int(round(float(durv)))} menit" if pd.notna(durv) else "-"
            dl = pd.Timestamp(r["max_date"]).strftime("%Y-%m-%d") if pd.notna(r.get("max_date")) else "-"
            markers.append({"lat": float(r["lat"]), "lon": float(r["lon"]), "well": str(r["well"]),
                            "cat": cat, "dur": dur, "unit": str(u), "seq": seq, "color": col, "deadline": dl})
            pts.append([float(r["lat"]), float(r["lon"])])
        if len(pts) > 1:
            routes.append({"coords": pts, "color": col})
    data = _json.dumps({"markers": markers, "routes": routes})
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(str(title))}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>"
        "<style>html,body,#map{height:100%;margin:0}"
        ".lbl{background:rgba(255,255,255,.85);border:0;border-radius:3px;padding:1px 4px;"
        "font:600 11px system-ui,sans-serif;color:#111;box-shadow:0 1px 2px rgba(0,0,0,.3)}</style>"
        "</head><body><div id='map'></div>"
        "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>"
        "<script>const DATA=" + data + ";"
        "const map=L.map('map');"
        "L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',"
        "{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(map);"
        "DATA.routes.forEach(r=>L.polyline(r.coords,{color:r.color,weight:3,opacity:.8}).addTo(map));"
        "const b=[];DATA.markers.forEach(m=>{"
        "const mk=L.circleMarker([m.lat,m.lon],{radius:7,color:'#fff',weight:1,fillColor:m.color,fillOpacity:.95}).addTo(map);"
        "mk.bindTooltip(m.seq+'. '+m.well,{permanent:true,direction:'right',className:'lbl'});"
        "mk.bindPopup('<b>'+m.well+'</b><br>'+m.cat+' · '+m.dur+'<br>Unit: '+m.unit+'<br>Deadline: '+m.deadline);"
        "b.push([m.lat,m.lon]);});"
        "if(b.length)map.fitBounds(b,{padding:[30,30]});else map.setView([1.6,101.3],9);"
        "</script></body></html>")

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
def plan(elig, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, current_day=None, elastic_limit=5.0, blocked_units=None, prebooked=None, min_wells=1, anchors=None):
    df = elig.reset_index(drop=True).copy()
    df["scheduled"] = False
    df["plan_unit"] = None
    df["is_seed"] = False   
    if "forced_unit" not in df.columns: df["forced_unit"] = None
    if "urgency" not in df.columns: df["urgency"] = 0.0
    df["_pre_unit"] = None

    if 'audit_logs' not in st.session_state:
        st.session_state['audit_logs'] = {}
    day_str = str(current_day)[:10] if current_day else "Hari_Ini"
    if day_str not in st.session_state['audit_logs']:
        st.session_state['audit_logs'][day_str] = []

    if anchors:
        _a = df["well"].map(lambda w: anchors.get(w))
        _ok = _a.notna()
        if "forced_unit" in df.columns:
            _fu = df["forced_unit"]
            _ok &= (_fu.isna() | (_fu == _a))
        df.loc[_ok, "_pre_unit"] = _a[_ok]
    if prebooked is not None and len(prebooked):
        pb = prebooked.copy()
        pb["_pre_unit"] = pb["plan_unit"]
        pb["forced_unit"] = None
        for _c in df.columns:
            if _c not in pb.columns: pb[_c] = None
        df = pd.concat([df, pb[df.columns]], ignore_index=True)

    lats = pd.to_numeric(df["lat"], errors="coerce").values
    lons = pd.to_numeric(df["lon"], errors="coerce").values
    dist_mat = _dist_matrix(lats, lons)

    field_arr = df["field"].values
    area_arr = df["area"].values
    has_fu = df["forced_unit"].notna().values
    fu_arr = df["forced_unit"].values
    # Mode Mapping Unit: allow_units = tuple unit yang berhak menggarap sumur ini.
    # Kosong/None → tak ada aturan, sumur ikut aturan zona remote/non-remote spt biasa.
    allow_arr = (df["allow_units"].values if "allow_units" in df.columns
                 else np.array([None] * len(df), dtype=object))

    def _unit_ok(i, u):
        """Bolehkah unit u mengambil sumur baris i? Mapping menang atas zona:
        kalau lapangan sudah dipetakan, daftar itulah kebenarannya."""
        allow = allow_arr[i]
        if allow:
            return u in allow
        return (area_arr[i] in REMOTE_AREAS) == (u in REMOTE_UNITS)

    urg_arr = pd.to_numeric(df["urgency"], errors="coerce").fillna(0).values
    dur_arr = pd.to_numeric(df["dur"], errors="coerce").fillna(0).values
    speed = max(float(speed), 1.0)

    _EL_BASE = elastic_limit
    _blk = set(blocked_units) if blocked_units else set()
    avail_remote = [u for u in list(REMOTE_UNITS)[:n_remote] if u not in _blk]
    avail_nonremote = [u for u in list(NONREMOTE_UNITS)[:n_nonremote] if u not in _blk]
    unit_clusters = {u: [] for u in avail_remote + avail_nonremote}
    unassigned = set(df.index)

    def _grow(u, target_fld=None, elim=None):
        elastic_limit = elim if elim is not None else _EL_BASE
        while len(unit_clusters[u]) < max_wells and unassigned:
            cand_pool = [i for i in unassigned if _unit_ok(i, u)
                         and not (has_fu[i] and fu_arr[i] != u)]
            if not cand_pool: break
            c_dists = dist_mat[np.ix_(unit_clusters[u], cand_pool)]
            min_dists = c_dists.min(axis=0)
            max_dists = c_dists.max(axis=0)
            valid_cands = []
            for i_cand, cand_idx in enumerate(cand_pool):
                d_min = min_dists[i_cand]; d_max = max_dists[i_cand]; urg = urg_arr[cand_idx]
                is_same_fld = (target_fld is not None and field_arr[cand_idx] == target_fld)
                if d_max > elastic_limit: continue
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

    # ── KLAUSA RELAKSASI UNTUK FORCED UNIT (GP & MPAS_525/768) ──
    # Cari tahu seberapa krisis hari ini secara global (nilai urgency paling kecil).
    # Add Manual diabaikan dalam penentuan urgensi global karena nilainya 1.000.000
    _urg_real = [urg_arr[i] for i in unassigned if urg_arr[i] < ADDMAN_URG]
    current_min_urg = min(_urg_real) if _urg_real else 0

    fu_mask = df["forced_unit"].notna() & df["forced_unit"].isin(unit_clusters.keys())
    for u, grp in df[fu_mask].groupby("forced_unit"):
        grp_sorted = grp.sort_values(["urgency", "dur"])
        for idx in grp_sorted.index:
            urg_idx = urg_arr[idx]
            
            # REVISI: Bebaskan unit jika sumur paksaannya tidak mendesak (H-3 ke atas) 
            # DAN di lapangan lain masih ada sumur yang krisisnya lebih tinggi dari dia.
            # Ini mewujudkan aturan: "Jika GP tidak mendesak, unit bisa dipakai untuk yg lain".
            if urg_idx > 2 and urg_idx > current_min_urg + 1:
                continue # Di-skip! GP ini gak akan narik unitnya sekarang. Nunggu ditarik saat _grow() atau besok.

            if len(unit_clusters[u]) < max_wells and idx in unassigned:
                if unit_clusters[u]:
                    d_min = dist_mat[unit_clusters[u], idx].min()
                    d_max = dist_mat[unit_clusters[u], idx].max()
                    if d_max > elastic_limit: continue
                    if d_min > 5.0 and not (d_min <= elastic_limit and urg_idx <= 2): continue
                if use_dur and unit_clusters[u]:
                    cand = unit_clusters[u] + [idx]
                    dist = route_distance(lats[cand], lons[cand])
                    if dur_arr[cand].sum() + (dist / speed) * 60 > time_budget: continue
                
                was_empty = (len(unit_clusters[u]) == 0)
                unit_clusters[u].append(idx)
                unassigned.remove(idx)
                
                if was_empty:
                    log_entry = {
                        "Unit": u, "Field Pemenang": field_arr[idx],
                        "Alasan Field": "Penugasan Mutlak (Fasilitas Khusus GP / MPAS_525).",
                        "Sumur Anchor": df.loc[idx, 'well'],
                        "Alasan Anchor": f"Dikunci ke {u}. Urgensi H-{int(urg_idx)} (Cukup Mendesak untuk mengamankan unit ini)."
                    }
                    if log_entry not in st.session_state['audit_logs'][day_str]:
                        st.session_state['audit_logs'][day_str].append(log_entry)

    _pre_units = set()
    if df["_pre_unit"].notna().any():
        for idx in df.index[df["_pre_unit"].notna()]:
            u = df.at[idx, "_pre_unit"]
            if not _unit_ok(idx, u): continue
            if u in unit_clusters and idx in unassigned:
                was_empty = (len(unit_clusters[u]) == 0)
                unit_clusters[u].append(idx); unassigned.discard(idx); _pre_units.add(u)
                
                if was_empty:
                    urg_seed = urg_arr[idx]
                    log_entry = {
                        "Unit": u, "Field Pemenang": field_arr[idx],
                        "Alasan Field": "Lapis 1 / Anchor Manual (Prioritas Sistem/User).",
                        "Sumur Anchor": df.loc[idx, 'well'],
                        "Alasan Anchor": f"Di-carry over dari Lapis 1 (NW/AWS/Req) atau ditunjuk manual."
                    }
                    if log_entry not in st.session_state['audit_logs'][day_str]:
                        st.session_state['audit_logs'][day_str].append(log_entry)

    for _u in [u for u, c in unit_clusters.items() if c]:
        _grow(_u)

    used_units = {u for u, c in unit_clusters.items() if len(c) > 0}
    avail_remote = [u for u in avail_remote if u not in used_units]
    avail_nonremote = [u for u in avail_nonremote if u not in used_units]

    _thin_fields = set()
    while unassigned and (avail_remote or avail_nonremote):
        field_scores = {}
        un_list = [i for i in unassigned if not has_fu[i]]
        if not un_list: break
        un_fields = field_arr[un_list]

        for fld in pd.unique(un_fields):
            f_wells = [w for w in un_list if field_arr[w] == fld]
            if not f_wells: continue
            urgs = urg_arr[f_wells]
            score = 0
            for u in urgs:
                if u >= ADDMAN_URG: score += 1
                elif u < 0: score += 100000
                elif u == 0: score += 50000
                elif u == 1: score += 10000
                elif u == 2: score += 5000
                elif u <= 4: score += 1000
                elif u <= 7: score += 100
                else: score += 10
            field_scores[fld] = score

        if not field_scores: break
        sorted_fields = sorted(field_scores.keys(), key=lambda k: field_scores[k], reverse=True)

        assigned_this_round = False
        for target_fld in sorted_fields:
            if target_fld in _thin_fields: continue
            f_wells = [w for w in unassigned if field_arr[w] == target_fld and not has_fu[w]]
            if not f_wells: continue
            f_wells_sorted = sorted(f_wells, key=lambda x: (urg_arr[x], dur_arr[x]))
            seed = f_wells_sorted[0]
            zone = "remote" if area_arr[seed] in REMOTE_AREAS else "nonremote"

            # Unit pembuka klaster harus yang berhak atas sumur seed. Tanpa mapping
            # pilihannya sebatas pool zonanya; dengan mapping, daftar unit lapangan
            # itulah yang menentukan — termasuk bila unitnya lintas zona.
            avail_pool = avail_remote if zone == "remote" else avail_nonremote
            if allow_arr[seed]:
                # Urutan unit di sheet = urutan pilihan. "MP445, MP523" berarti 445 dicoba dulu.
                _av = avail_remote + avail_nonremote
                u = next((x for x in allow_arr[seed] if x in _av), None)
                if u is None: continue
                avail_pool = avail_remote if u in REMOTE_UNITS else avail_nonremote
                avail_pool.remove(u)
            else:
                if not avail_pool: continue
                u = avail_pool.pop(0)
            
            # ── X-RAY LOG: MENCARI SAINGAN DI ZONA YANG SAMA ──
            saingan_list = []
            for f in sorted_fields:
                if f == target_fld: continue
                _fw = [w for w in unassigned if field_arr[w] == f and not has_fu[w]]
                if _fw and ("remote" if area_arr[_fw[0]] in REMOTE_AREAS else "nonremote") == zone:
                    saingan_list.append(f"{f} ({field_scores[f]} pts)")
                if len(saingan_list) >= 3: break
            
            saingan_str = " | ".join(saingan_list) if saingan_list else "Tidak ada saingan di zona ini"
            
            urg_seed = urg_arr[seed]
            log_entry = {
                "Unit": u, "Field Pemenang": target_fld,
                "Alasan Field": f"Skor tertinggi di area {zone} ({field_scores[target_fld]} pts). Saingan se-zona: {saingan_str}",
                "Sumur Anchor": df.loc[seed, 'well'],
                "Alasan Anchor": f"Urgensi (H-{int(urg_seed)}) & Durasi ({dur_arr[seed]} min) terbaik di {target_fld}."
            }
            if log_entry not in st.session_state['audit_logs'][day_str]:
                st.session_state['audit_logs'][day_str].append(log_entry)
            # ──────────────────────────────────────────────────

            unit_clusters[u].append(seed)
            unassigned.remove(seed)
            used_units.add(u)

            _grow(u, target_fld)

            if min_wells > 1 and len(unit_clusters[u]) < min_wells:
                _grow(u, target_fld, elim=_EL_BASE * 1.8)
            if min_wells > 1 and len(unit_clusters[u]) < min_wells \
               and not any(urg_arr[i] <= 0 for i in unit_clusters[u]):
                unassigned.update(unit_clusters[u])
                unit_clusters[u] = []
                used_units.discard(u)
                avail_pool.insert(0, u)
                _thin_fields.add(target_fld)
                
                st.session_state['audit_logs'][day_str] = [log for log in st.session_state['audit_logs'][day_str] if log["Unit"] != u]
                continue

            assigned_this_round = True
            break
        if not assigned_this_round: break

    for u, c in unit_clusters.items():
        if c:
            df.loc[c, "scheduled"] = True
            df.loc[c, "plan_unit"] = u
            df.loc[c[0], "is_seed"] = True

    return df

# ── Grouping ulang murni berdasarkan kedekatan sumur ───────────────────────
# Aturan deadline yang memutuskan sumur mana digarap HARI apa dan pakai BERAPA unit.
# Setelah itu keanggotaan klaster masih menyimpan jejak urutan penjadwalan: sumur yang
# kebetulan diproses belakangan bisa nyangkut di unit yang rutenya jadi memutar.
# Fungsi ini menyusun ULANG keanggotaan itu murni dari kedekatan geografis, TANPA
# mengubah hari, jumlah unit, maupun daftar sumur yang terjadwal — jadi kepatuhan
# deadline hasil run sebelumnya tetap utuh, yang berubah hanya siapa berangkat bersama siapa.
def regroup_by_proximity(wk, max_wells, iters=8):
    """Return (week_df_baru, info). Bekerja per (hari, zona); klaster yang tak bisa
    disusun ulang tanpa melanggar batasan dibiarkan apa adanya."""
    wk = wk.copy()
    info = {"grup": 0, "pindah": 0, "km_awal": 0.0, "km_akhir": 0.0, "gagal": 0, "tak_untung": 0}
    if not len(wk) or "plan_unit" not in wk.columns:
        return wk, info

    lat = pd.to_numeric(wk.get("lat"), errors="coerce")
    lon = pd.to_numeric(wk.get("lon"), errors="coerce")
    coord_ok = lat.notna() & lon.notna()
    manual = wk["manual"].fillna(False) if "manual" in wk.columns else pd.Series(False, index=wk.index)
    urg = pd.to_numeric(wk.get("urgency"), errors="coerce").fillna(0)

    def boleh(i, u):
        """Batasan keras tetap dihormati: forced_unit (GP/fasilitas) & mapping unit."""
        fu = wk.at[i, "forced_unit"] if "forced_unit" in wk.columns else None
        if fu is not None and not (isinstance(fu, float) and pd.isna(fu)) and str(fu) != "None" and fu != u:
            return False
        au = wk.at[i, "allow_units"] if "allow_units" in wk.columns else None
        if isinstance(au, (list, tuple)) and len(au) and u not in au:
            return False
        return True

    def km_of(idxs):
        """Total km rute satu klaster, memakai kalkulator rute yang sama dgn seluruh app."""
        if len(idxs) < 2:
            return 0.0
        return route_distance(lat.loc[idxs].values, lon.loc[idxs].values)

    sched = wk["scheduled"].fillna(False) if "scheduled" in wk.columns else pd.Series(False, index=wk.index)
    if "zone" in wk.columns:
        zone = wk["zone"]
    else:
        zone = np.where(wk["area"].isin(REMOTE_AREAS), "remote", "non-remote")
        zone = pd.Series(zone, index=wk.index)

    for (_di, _z), blok in wk[sched].groupby([wk.loc[sched, "day_idx"], zone[sched]]):
        idxs = list(blok.index)
        units = [u for u in pd.unique(blok["plan_unit"]) if u is not None and pd.notna(u)]
        awal = {u: [i for i in idxs if wk.at[i, "plan_unit"] == u] for u in units}
        _km_awal_blok = sum(km_of(v) for v in awal.values())
        info["km_awal"] += _km_awal_blok
        bebas = [i for i in idxs if coord_ok[i] and not manual[i]]
        _bebas_set = set(bebas)
        # Apakah susunan ASLI sudah sah? (tiap sumur bebas ada di unit yang boleh).
        # Kalau TIDAK sah (mis. forced_unit/mapping dilanggar), penyusunan ulang WAJIB
        # dipakai untuk memperbaikinya, gerbang anti-boros km tak berlaku.
        _awal_sah = all(boleh(i, u) for u in units for i in awal[u] if i in _bebas_set)
        tetap = {u: [i for i in awal[u] if i not in bebas] for u in units}
        if len(units) < 2 or not bebas:
            info["km_akhir"] += sum(km_of(v) for v in awal.values())
            continue

        anggota = {u: list(awal[u]) for u in units}
        for _ in range(iters):
            pusat = {}
            for u in units:
                a = anggota[u]
                ok = [i for i in a if coord_ok[i]]
                pusat[u] = (lat.loc[ok].mean(), lon.loc[ok].mean()) if ok else (np.nan, np.nan)

            baru = {u: list(tetap[u]) for u in units}
            sisa = {u: int(max_wells) - len(baru[u]) for u in units}
            belum = set(bebas)

            # (a) sumur paling terkekang lebih dulu. Sumur ber-forced_unit atau ber-mapping
            #     sempit cuma punya satu unit yang sah; kalau ia menunggu giliran greedy,
            #     unit itu sudah keburu penuh oleh sumur bebas dan seluruh penyusunan ulang
            #     batal sia-sia — persis kasus yang bikin klaster ber-GP tak pernah dirapikan.
            for i in sorted(belum, key=lambda w: sum(1 for u in units if boleh(w, u))):
                sah = [u for u in units if boleh(i, u)]
                if len(sah) == 1 and sisa[sah[0]] > 0:
                    baru[sah[0]].append(i); belum.discard(i); sisa[sah[0]] -= 1

            # (b) unit yang masih kosong dijatah satu sumur terdekat supaya tak ada unit
            #     yang kehilangan seluruh muatannya (trip-nya batal) gara-gara penyusunan ulang.
            for u in units:
                if baru[u] or sisa[u] <= 0 or np.isnan(pusat[u][0]):
                    continue
                kand = [(haversine_km(lat[i], lon[i], *pusat[u]), i) for i in belum if boleh(i, u)]
                if kand:
                    _, i = min(kand)
                    baru[u].append(i); belum.discard(i); sisa[u] -= 1

            # (c) sisanya: pasangan (sumur, unit) terdekat lebih dulu, hormati kapasitas
            pasang = sorted((haversine_km(lat[i], lon[i], *pusat[u]), i, u)
                            for i in belum for u in units
                            if not np.isnan(pusat[u][0]) and boleh(i, u))
            for _d, i, u in pasang:
                if i in belum and sisa[u] > 0:
                    baru[u].append(i); belum.discard(i); sisa[u] -= 1

            if belum:            # ada yang tak kebagian → batalkan, pertahankan susunan asli
                anggota = None
                break
            if all(set(baru[u]) == set(anggota[u]) for u in units):
                anggota = baru
                break
            anggota = baru

        if anggota is None:
            info["gagal"] += 1
            info["km_akhir"] += _km_awal_blok
            continue

        # GERBANG ANTI-BOROS: penyusunan ulang cuma dipakai kalau rute blok ini TIDAK
        # bertambah panjang. Kalau malah menambah jarak, susunan asli dipertahankan.
        # (Assignment k-means meminimalkan jarak-ke-centroid, bukan panjang rute TSP,
        #  jadi ada kasus centroid rapi tapi rute lebih boros — di situ regroup dibatalkan.)
        _km_baru_blok = sum(km_of(v) for v in anggota.values())
        if _awal_sah and _km_baru_blok > _km_awal_blok + 1e-6:
            info["tak_untung"] += 1
            info["km_akhir"] += _km_awal_blok
            continue

        info["grup"] += 1
        info["km_akhir"] += _km_baru_blok
        for u in units:
            for i in anggota[u]:
                if wk.at[i, "plan_unit"] != u:
                    info["pindah"] += 1
                wk.at[i, "plan_unit"] = u
            # seed = sumur paling mendesak di klaster barunya (dipakai penanda di peta)
            if "is_seed" in wk.columns and anggota[u]:
                for i in anggota[u]:
                    wk.at[i, "is_seed"] = False
                wk.at[min(anggota[u], key=lambda i: urg[i]), "is_seed"] = True
    return wk, info

def build_elig(raw, ncmp_df, per_lo_ts, per_hi_ts, week_lo, week_hi,
               executed, comp_disp_set, pending_set, ncmp_replan, woff_set):
    """Kelayakan & urgensi satu periode: dari kandidat mentah → elig (siap dijadwalkan).
    Dijadikan fungsi supaya tab Analisis Performa bisa menjalankan periode LAIN dengan
    aturan yang persis sama — kalau blok ini disalin, dua salinannya pasti berbeda
    suatu hari dan angka analisis berhenti bisa dipercaya."""
    batch_lo, batch_hi = per_lo_ts, per_hi_ts
    win_in_range = (raw["min_date"] <= batch_hi) & (raw["max_date"] >= batch_lo)   # overlap window min-max

    # NCMP ber-COMMENT IF NOT COMPLETE = FACI/ROAD/WOFF → dibawa ulang sepanjang periode.
    # Kegagalannya hambatan lapangan, bukan kapasitas kru, jadi window yang sudah lewat
    # tidak boleh mementalkannya seperti carry-over NCMP biasa.
    ncmp_carry = {r.well: r.kode_hambatan for r in ncmp_df.itertuples()
                  if r.kode_hambatan and r.well in ncmp_replan}

    # Carry-over NCMP TIDAK menembus aturan overlap. NCMP dari periode lampau yang
    # window-nya sudah lewat = overdue, dan overdue hanya boleh utk PRQ/ORQ (+FACI/ROAD/WOFF).
    # NCMP yang window-nya masih overlap (mis. gagal di H3, dijadwal ulang H7) tetap jalan.
    _ncmp_ok = set(raw.loc[raw["well"].isin(ncmp_replan)
                           & (win_in_range | raw["well"].isin(ncmp_carry)), "well"])
    ncmp_expired = sorted(ncmp_replan - _ncmp_ok)
    ncmp_replan = _ncmp_ok
    replan_df = ncmp_df[ncmp_df["well"].isin(ncmp_replan)].copy()
    expired_df = raw[raw["well"].isin(ncmp_expired)][["well", "field", "area", "min_date", "max_date"]].copy()
    np_in_range = (raw["next_wt"] >= batch_lo) & (raw["next_wt"] <= batch_hi)      # next_proposed_wt di periode
    is_nwaws_c = raw["is_nwaws"].fillna(False)
    # PRQ boleh dijadwalkan DI LUAR window min-max HANYA jika diambil dari sheet Compiled
    # Schedule (is_breakin == False). PRQ yang berasal dari sheet BreakIn WAJIB tunduk pada
    # window min-max, jadi ia hanya eligible bila window-nya overlap periode (lewat jalur
    # win_in_range), bukan lewat req_force / overdue_prio. ORQ tidak terpengaruh aturan ini.
    is_breakin_c = raw["is_breakin"].fillna(False) if "is_breakin" in raw.columns else pd.Series(False, index=raw.index)
    prq_breakin = (raw["req_tag"] == "PRQ") & is_breakin_c
    # Kelayakan dasar = window min-max OVERLAP periode, utk SEMUA kategori (RTN/NW/AWS/Add Manual).
    # Jalur next_wt TIDAK membuat sumur non-overlap jadi eligible (mis. AWS1 BO497 window 30 Mei–1 Jun
    # tapi next_wt 22 Jun → tetap TIDAK eligible). PENGECUALIAN: PRQ/ORQ selalu boleh (via req_force /
    # overdue_prio), termasuk saat overdue — permintaan boleh dipenuhi sepanjang periode.
    in_range = win_in_range
    req_force = raw["force_week"].fillna(False) & ~is_nwaws_c & ~prq_breakin
    is_ncmp = raw["well"].isin(ncmp_replan)
    # OVERDUE (deadline sudah lewat sebelum awal periode) → HANYA PRQ/ORQ yang tetap
    # dijadwalkan (permintaan boleh dipenuhi sepanjang periode). NW/AWS yang window min–max-nya
    # SELURUHNYA di luar periode (tak overlap) TIDAK dijadwalkan — window-nya sudah terlewat,
    # bukan urusan periode ini (mis. AWS1 BO497 window 30 Mei–1 Jun utk periode 22–30 Jun).
    is_prio_c = is_nwaws_c | raw["req_tag"].isin(["PRQ", "ORQ"])
    overdue_prio = raw["req_tag"].isin(["PRQ", "ORQ"]) & ~prq_breakin & raw["max_date"].notna() & (raw["max_date"] < batch_lo)
    # Add Manual: tetap eligible walau window sudah lewat, selama masih bisa dimulai dalam periode
    # (min_date <= batch_hi). Prioritas rendah diatur belakangan; di sini hanya soal kelayakan.
    _addman_c = raw["is_addmanual"].fillna(False) if "is_addmanual" in raw.columns else pd.Series(False, index=raw.index)
    # Add Manual pun HARUS overlap window periode (tak boleh overdue). Hanya PRQ/ORQ yg boleh overdue.
    addman_c = _addman_c & win_in_range
    comp_wells = raw[raw["well"].isin(comp_disp_set)].copy()
    pending_wells = raw[raw["well"].isin(pending_set)].copy()
    pending_nodata = sorted(pending_set - set(pending_wells["well"]))
    nwaws_dropped = raw[is_nwaws_c & ~in_range & ~overdue_prio & (~raw["well"].isin(executed))].copy()
    cand = raw[(in_range | is_ncmp | req_force | overdue_prio | addman_c) & (~raw["well"].isin(executed | pending_set))].copy()

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

    # Slack: deadline (max_date) SETELAH akhir periode → NW/AWS tak wajib dites sekarang, boleh ditunda.
    # NW/AWS ber-slack tidak diberi boost prioritas — jadi pengisi celah, tak menyerobot sumur yg
    # deadline-nya jatuh di dalam periode. PRQ/ORQ (+NCMP) = mid_prio DIKECUALIKAN (boleh sepanjang periode).
    _slack_e = elig_all["max_date"].notna() & (elig_all["max_date"] > batch_hi)
    elig_all.loc[mid_prio, "urgency"] = elig_all.loc[mid_prio, "urgency"].clip(upper=0)
    elig_all.loc[nwaws & ~_slack_e, "urgency"] = elig_all.loc[nwaws & ~_slack_e, "urgency"].clip(upper=0) - 10000

    # Sumur REGULER (bukan NW/AWS/PRQ/ORQ/carry-NCMP) yang window min-max-nya di LUAR rentang periode
    # → urgensi FLEKSIBEL: tak wajib dites di awal, boleh kapan saja dalam rentang (isi celah).
    # (Kebalikan sumur prioritas overdue yang justru harus paling dulu.)
    _prio_u = nwaws | mid_prio | elig_all["req_tag"].isin(["PRQ", "ORQ"])
    _win_outside = (elig_all["max_date"] < batch_lo) | (elig_all["min_date"] > batch_hi)
    _flex = (~_prio_u) & elig_all["max_date"].notna() & _win_outside
    _span = max(int((week_hi - week_lo).days), 1)
    elig_all.loc[_flex, "urgency"] = _span
    # Add Manual → urgensi fleksibel (isi celah). Prioritas terendah sesungguhnya diterapkan
    # ulang per-hari di plan_week (ADDMAN_URG), agar tidak menggeser sumur lain.
    _addman_u = elig_all["is_addmanual"].fillna(False) if "is_addmanual" in elig_all.columns else pd.Series(False, index=elig_all.index)
    elig_all.loc[_addman_u & ~_prio_u, "urgency"] = _span

    # NCMP FACI/ROAD/WOFF yang deadline-nya SUDAH lewat: tak ada gunanya diborong di hari
    # pertama — deadline-nya toh sudah terlewat. Urgensinya dibuat fleksibel supaya ia
    # dijadwalkan ulang di sepanjang sisa periode, mengisi celah rute tanpa menggeser
    # sumur yang deadline-nya masih hidup. Yang deadline-nya masih di dalam periode tetap
    # ikut mid_prio (urgensi asli) di atas.
    elig_all["carry_code"] = elig_all["well"].map(ncmp_carry).fillna("")
    _carry_late = elig_all["carry_code"].astype(bool) & elig_all["max_date"].notna() & (elig_all["max_date"] < week_lo)
    elig_all.loc[_carry_late, "urgency"] = _span

    elig = elig_all[elig_all["has_coord"]].copy()
    nocoord = elig_all[~elig_all["has_coord"]].copy()

    return {"batch_lo": batch_lo, "batch_hi": batch_hi, "cand": cand, "comp_wells": comp_wells,
            "elig": elig, "elig_all": elig_all, "expired_df": expired_df, "ncmp_carry": ncmp_carry,
            "ncmp_expired": ncmp_expired, "ncmp_replan": ncmp_replan, "nocoord": nocoord,
            "off_wells": off_wells, "pending_nodata": pending_nodata, "pending_wells": pending_wells,
            "replan_df": replan_df, "woff_wells": woff_wells}


# ── Analisis Performa lintas periode ───────────────────────────────────────
# Kolom laju minyak di sheet kandidat dipakai untuk menaksir Lost Oil. Namanya bebas
# selama mengandung salah satu kata kunci ini; kalau tak ada satu pun, kolom Lost Oil
# dikosongkan dan bukan ditebak.
OIL_COL_HINTS = ("BOPD", "NET_OIL", "NET OIL", "OIL_RATE", "OIL RATE", "OIL_PROD", "LAST_OIL", "OIL")

def find_oil_col(df):
    up = {str(c).strip().upper(): c for c in df.columns}
    for h in OIL_COL_HINTS:
        for u, c in up.items():
            if h in u and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce")):
                return c
    return None

def bench_period(r, per_lo, per_hi, umap, params, oil_col=None):
    """Jadwalkan ulang SATU periode dari nol untuk satu mode, lalu ukur compliance & km/well.

    Sengaja TANPA status realisasi (COMP/NCMP/PENDING) dari SCH_Database. Kalau hasil periode
    itu diintip, sumur yang dulu memang sudah dites akan dikeluarkan dari kandidat dan WELLGO
    seolah tak punya pekerjaan — angkanya jadi tak bermakna. Tiap periode dinilai apa adanya
    dari pool kandidatnya, sama seperti perencana manual saat periode itu belum berjalan."""
    lo, hi = pd.Timestamp(per_lo), pd.Timestamp(per_hi)
    days = [lo + pd.Timedelta(days=i) for i in range(max(int((hi - lo).days), 0) + 1)]
    r = r.copy()
    r["allow_units"] = unit_map_allow(r, umap) if umap else None
    _empty_ncmp = pd.DataFrame(columns=["well", "kode_hambatan"])
    E = build_elig(r, _empty_ncmp, lo, hi, days[0], days[-1],
                   set(), set(), set(), set(), set())
    elig, nocoord = E["elig"], E["nocoord"]
    if not len(elig) and not len(nocoord):
        return None

    # plan_week mereset st.session_state['audit_logs'] tiap dipanggil; jalannya analisis tak
    # boleh menghapus jejak audit run utama yang sedang ditampilkan di tab lain.
    _audit_backup = st.session_state.get("audit_logs", {})
    try:
        wk = plan_week(elig, days, "pooled", params["max_wells"], params["n_remote"],
                       params["n_nonremote"], params["time_budget"], params["speed"],
                       params["use_urg"], params["use_dur"], early_days=params["early_days"],
                       elastic_limit=params["elastic_limit"], min_wells=params["min_wells"])
    finally:
        st.session_state["audit_logs"] = _audit_backup
    if len(nocoord):
        wk = pd.concat([wk, nocoord.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)],
                       ignore_index=True)

    sched = wk[wk["scheduled"]]
    km = 0.0
    for _, sub in sched.groupby(["day_idx", "plan_day", "plan_unit"]):
        c = sub[sub["has_coord"].fillna(False)]
        km += route_distance(c["lat"].values, c["lon"].values) if len(c) > 1 else 0.0
    n_coord = int(sched["has_coord"].fillna(False).sum()) if len(sched) else 0

    # ── Compliance BERBASIS ON-TIME ──────────────────────────────────────────
    # Populasi = sumur yang deadline (max_date) jatuh DI DALAM periode ("due").
    # Comply  = terjadwal pada hari <= deadline (on-time).
    # Not-Comply = LATE (terjadwal tapi hari > deadline) ATAU MISS (tak terjadwal).
    due = wk[wk["max_date"].between(lo, hi)].copy()
    _sc = due["scheduled"].fillna(False)
    _pday = pd.to_datetime(due["plan_day"], errors="coerce")
    # PRQ/ORQ (request prioritas) dikecualikan seperti WELLGO: sekali terjadwal, kapan pun
    # tanggalnya, dihitung on-time (tak pernah Late). Yang tak terjadwal tetap Miss.
    _prq = (due["req_tag"].isin(["PRQ", "ORQ"]) if "req_tag" in due.columns
            else pd.Series(False, index=due.index))
    _late = _sc & _pday.notna() & (_pday > due["max_date"]) & ~_prq
    _ontime = _sc & ~_late                    # terjadwal & bukan late → on-time
    _miss = ~_sc
    comply = int(_ontime.sum())
    denom = len(due)
    miss = due[_miss]                         # tak terjadwal → dipakai utk lost oil
    if oil_col and oil_col in wk.columns:
        lost_oil = float(pd.to_numeric(miss[oil_col], errors="coerce").fillna(0).sum()) if len(miss) else 0.0
    else:
        lost_oil = np.nan
    # Daftar Not-Comply (LATE + MISS) beserta window jadwal & tanggal jadwalnya.
    _mc = [c for c in ("well", "field", "area", "subarea", "category", "unit",
                       "min_date", "max_date", "urgency") if c in due.columns]
    miss_df = due[_late | _miss][_mc].copy()
    miss_df["status"] = np.where(_late[_late | _miss].values, "Late (lewat deadline)", "Miss (tak terjadwal)")
    miss_df["sched_date"] = _pday[_late | _miss].where(_late[_late | _miss]).values
    return {"kandidat": len(wk), "terjadwal": len(sched), "on_time": comply,
            "late": int(_late.sum()), "miss": int(_miss.sum()),
            "compliance": (100.0 * comply / denom) if denom else 100.0,
            "km": km, "km_well": (km / n_coord) if n_coord else 0.0, "lost_oil": lost_oil,
            "miss_df": miss_df,
            # Pool kandidat periode ini beserta deadline & atributnya — dipakai sisi Manual
            # supaya penyebut compliance-nya sama persis, dan daftar Not-Comply manual bisa
            # ikut membawa field/area/window jadwal seperti daftar Not-Comply WELLGO.
            "pool": wk[[c for c in ("well", "max_date", "min_date", "field", "area",
                                    "subarea", "category", "unit") if c in wk.columns]].copy()}

def bench_manual(hist, per_lo, per_hi, coord_map, pool):
    """Compliance BERBASIS ON-TIME & km/well jadwal manual satu periode dari file history.

    Deadline sumur TIDAK ada di file history, jadi diambil dari pool kandidat periode ini —
    pool yang sama yang dipakai menilai WELLGO, supaya kedua sisi punya penyebut identik.
    Populasi = sumur kandidat yang deadline-nya jatuh di dalam periode ("due"). Comply =
    sumur tsb dites manual pada tanggal <= deadline (on-time). Not-Comply = LATE (dites
    tapi setelah deadline) ATAU MISS (tak muncul di history sama sekali). Baris history di
    luar pool kandidat tak ikut dihitung — menjadwalkan sumur yang bukan tanggungan periode
    ini tak menutup deadline satu pun, dan memasukkannya menggelembungkan compliance."""
    lo, hi = pd.Timestamp(per_lo), pd.Timestamp(per_hi)
    h = hist[(hist["date"] >= lo) & (hist["date"] <= hi)].copy() if len(hist) else hist
    if not len(h):
        return None
    h["lat"] = h["well"].map(coord_map.get("lat", {}))
    h["lon"] = h["well"].map(coord_map.get("lon", {}))
    v = h[h["lat"].notna() & h["lon"].notna()]
    km = sum(route_distance(g["lat"].values, g["lon"].values)
             for _, g in v.groupby([v["date"].dt.date, "unit"]) if len(g) > 1) if len(v) else 0.0

    # Tanggal tes manual per sumur = tanggal TERAWAL ia muncul di history dalam periode.
    hmin = h.groupby("well")["date"].min()
    # PRQ/ORQ pada manual dikenali dari kolom REASON (format SCHDatabase). Sekali dites,
    # kapan pun tanggalnya, dihitung on-time (tak pernah Late) — sama seperti sisi WELLGO.
    if "reason" in h.columns:
        _rprq = h["reason"].astype(str).str.upper().str.contains("PRQ|ORQ", na=False, regex=True)
        prq_wells = set(h.loc[_rprq, "well"])
    else:
        prq_wells = set()
    due = pool[pool["max_date"].between(lo, hi)].copy()
    due["_tested"] = due["well"].map(hmin)
    _prq = due["well"].isin(prq_wells)
    _late = due["_tested"].notna() & (due["_tested"] > due["max_date"]) & ~_prq
    _ontime = due["_tested"].notna() & ~_late
    _miss = due["_tested"].isna()
    comply = int(_ontime.sum())
    denom = len(due)
    _mc = [c for c in ("well", "field", "area", "subarea", "category", "unit",
                       "min_date", "max_date") if c in due.columns]
    miss_df = due[_late | _miss][_mc].copy()
    miss_df["status"] = np.where(_late[_late | _miss].values, "Late (lewat deadline)", "Miss (tak terjadwal)")
    miss_df["sched_date"] = due["_tested"][_late | _miss].where(_late[_late | _miss]).values
    return {"wells": len(v), "luar_pool": int(h["well"].nunique() - len(set(h["well"]) & set(pool["well"]))),
            "km": km, "km_well": (km / len(v)) if len(v) else np.nan,
            "on_time": comply, "late": int(_late.sum()), "miss": int(_miss.sum()), "miss_df": miss_df,
            "compliance": (100.0 * comply / denom) if denom else np.nan}

def plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed,
              use_urg, use_dur, early_days=0, elastic_limit=5.0, unit_blackout=None, prebooked=None, day_offset=0,
              min_wells=1, anchors=None, plan_fn=None):
    # plan_fn = mesin penjadwal per-hari (default heuristik plan()).
    plan_fn = plan_fn or plan
              
    # ── RESET TRACKER SAAT RUN BARU (Anti Numpuk) ──
    if prebooked is None:
        st.session_state['audit_logs'] = {}
    # ───────────────────────────────────────────────
    
    elig = elig.reset_index(drop=True).copy()
    elig["scheduled"] = False
    elig["plan_unit"] = None
    elig["plan_day"] = pd.NaT
    elig["day_idx"] = 0
    elig["is_seed"] = False
    early_td = pd.Timedelta(days=early_days)
    rem = pd.Series(True, index=elig.index)

    is_nwaws = elig["tipe"].isin(["NW", "AWS"])
    is_addman = elig["is_addmanual"].fillna(False) if "is_addmanual" in elig.columns else pd.Series(False, index=elig.index)
    fw_c = elig["force_week"].fillna(False) if "force_week" in elig.columns else pd.Series(False, index=elig.index)
    cc_c = elig["carry_ncmp"].fillna(False) if "carry_ncmp" in elig.columns else pd.Series(False, index=elig.index)
    
    bypass_reg = (fw_c & ~is_nwaws) | (cc_c & ~is_nwaws)

    np_in = elig["np_in_range"].fillna(False) if "np_in_range" in elig.columns else pd.Series(False, index=elig.index)
    next_wt = elig["next_wt"] if "next_wt" in elig.columns else pd.Series(pd.NaT, index=elig.index)
    
    strict_no_late = elig["np_in_range"] & elig["max_in_range"] & ~is_nwaws
    overdue_nw = is_nwaws & elig["max_date"].notna() & (elig["max_date"] < days[0])

    _is_reg_a = elig["is_reg_a"].fillna(False) if "is_reg_a" in elig.columns else pd.Series(False, index=elig.index)
    _early_row = pd.Series(early_td, index=elig.index)
    _early_row[_is_reg_a] = pd.Timedelta(0)

    for i, day in enumerate(days, start=1):
        win_reg = (elig["min_date"] - _early_row <= day)
        win_nw = (elig["min_date"] <= day) & (elig["max_date"] >= day)
        np_ok = np_in & (next_wt <= day) & ~is_nwaws

        is_late = day > elig["max_date"]
        forbid_late = strict_no_late & is_late

        cond_reg = (~is_nwaws) & (win_reg | np_ok | bypass_reg) & ~forbid_late
        cond_nw = is_nwaws & (win_nw | overdue_nw)
        
        pidx = elig.index[rem & (cond_reg | cond_nw)]
        if len(pidx) == 0: continue

        pool = elig.loc[pidx].copy()
        # Urgensi dasar berdasarkan kalender asli
        pool["urgency"] = (pool["max_date"] - day).dt.days.fillna(0)
        
        mid = bypass_reg.loc[pidx]
        nw = is_nwaws.loc[pidx]

        _period_end = pd.Timestamp(days[-1])
        _slack = pool["max_date"].notna() & (pool["max_date"] > _period_end)
        
        # ── REVISI: PRQ/ORQ Ngalah ke H-0 (Mencari Hari Longgar) ──
        # Kita hapus clip(upper=0).
        # Jika PRQ/ORQ ini overdue atau H-0 (urgency <= 0), kita set urgensinya menjadi 1 (H-1).
        # Efek: PRQ akan dapat skor lapangan 10.000, KALAH dari reguler H-0 yang dapat 50.000.
        # Tapi saat di hari tersebut tidak ada H-0 (hari longgar), PRQ akan langsung dieksekusi.
        is_prq_orq = mid
        pool.loc[is_prq_orq & (pool["urgency"] <= 0), "urgency"] = 1
        # ──────────────────────────────────────────────────────────

        pool.loc[is_addman.loc[pidx], "urgency"] = ADDMAN_URG

        day_key = pd.Timestamp(day).strftime("%Y-%m-%d")
        blocked = unit_blackout.get(day_key, set()) if unit_blackout else set()
        if blocked and "forced_unit" in pool.columns:
            pool = pool[~(pool["forced_unit"].notna() & pool["forced_unit"].isin(blocked))]
            if len(pool) == 0: continue

        _anch_day = {w: a["unit"] for w, a in (anchors or {}).items()
                     if int(a.get("day_idx", 0)) == (i + day_offset)} or None

        pb_day = None
        if prebooked is not None and len(prebooked):
            _pbd = prebooked[prebooked["day_idx"] == (i + day_offset)]
            if len(_pbd): pb_day = _pbd

        pd_ = plan_fn(pool, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, current_day=day, elastic_limit=elastic_limit, blocked_units=blocked, prebooked=pb_day, min_wells=min_wells, anchors=_anch_day)

        sd = pd_[pd_["scheduled"]]
        if len(sd) == 0: continue

        sidx = elig.index[elig["well"].isin(sd["well"])]
        elig.loc[sidx, "scheduled"] = True
        elig.loc[sidx, "plan_day"] = day
        elig.loc[sidx, "day_idx"] = i + day_offset
        elig.loc[sidx, "plan_unit"] = elig.loc[sidx, "well"].map(dict(zip(sd["well"], sd["plan_unit"])))
        if "is_seed" in sd.columns:
            elig.loc[sidx, "is_seed"] = elig.loc[sidx, "well"].map(dict(zip(sd["well"], sd["is_seed"]))).fillna(False)
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

# ── Sheet multi-periode (Compiled Schedule): kolom "Rentang Periode" + Start/End Periode ──
# Bila ada, WELLGO hanya memproses baris pada rentang periode yang DIPILIH user.
_pcol = next((c for c in raw.columns if str(c).strip().lower() in ("rentang periode", "rentang_periode")), None)
_pscol = next((c for c in raw.columns if "start" in str(c).lower() and "period" in str(c).lower()), None)
_pecol = next((c for c in raw.columns if str(c).lower().startswith("end") and "period" in str(c).lower()), None)
HAS_PERIODS = bool(_pcol and _pscol)
period_opts = None
if HAS_PERIODS:
    raw["_rperiode"] = raw[_pcol].astype(str).str.strip().replace({"nan": np.nan, "": np.nan, "None": np.nan})
    raw["_pstart"] = pd.to_datetime(raw[_pscol], errors="coerce")
    raw["_pend"] = pd.to_datetime(raw[_pecol], errors="coerce") if _pecol else pd.NaT
    period_opts = (raw.dropna(subset=["_rperiode", "_pstart"])
                      .groupby("_rperiode").agg(_s=("_pstart", "min"), _e=("_pend", "max"))
                      .sort_values("_s"))

spatial_db = load_spatial_data(up.getvalue(), sheet_spasial)

if spatial_db.empty: st.error("⚠️ Struktur berkas Data Spasial tidak valid atau kosong. Pastikan sheet mengandung kolom: WELL, FIELD, LAT, LON.")

# Meneruskan Sidebar...
with st.sidebar:
    st.divider()
    ui.section("🗺️ Filter Area & Unit")
    all_areas = sorted(raw["area"].dropna().unique())
    default_excl = [a for a in all_areas if a == "LIBO"]
    excl_areas = st.multiselect("Exclude Area Terpilih", all_areas, default=default_excl)
    default_tsdown = [a for a in all_areas if a in ["BANGKO", "BALAM"]]
    if mpas_only:
        ts_unavail = st.multiselect("Area Fasilitas TS Down (Dialihkan ke MWT)", all_areas,
                                    default=default_tsdown)
        mwt_unavail = st.multiselect("Area Fleet MWT Down (Dialihkan ke TS)", all_areas)
    else:
        ts_unavail, mwt_unavail = [], []
    
    st.divider()
    ui.section("⏱️ Status Realisasi Harian")
    comp_files = st.file_uploader("Upload file COMP/NCMP harian", type=["xlsx", "xlsm"], accept_multiple_files=True)
    skip_woff = st.checkbox("Skip sumur NCMP yang berstatus OFF", value=True,
        help="Berdasarkan Well Status di master kandidat, bukan kolom COMMENT IF NOT COMPLETE. "
             "Sumur NCMP ber-comment WOFF yang di master masih ON tetap dijadwalkan ulang; "
             "matikan centang ini bila yang berstatus OFF pun ingin dibawa ulang.")
    aws_split = st.checkbox("Pecah AWS jadi 2 kunjungan (AWS1 + AWS2) dalam periode", value=False,
        help="Untuk capacity planning: bila window AWS1 (POP+1..+3) DAN taksiran window AWS2 sama-sama "
             "masuk periode terpilih & AWS1 belum ada bukti selesai, sumur dibuat jadi 2 tugas terpisah sekaligus. "
             "Karena AWS1 belum dites, window AWS2 preview ditaksir relatif window AWS1. Begitu AWS1 di-COMP, "
             "window AWS2 dihitung ulang dari tanggal tes AWS1 (last_wt+5..+10) & alur balik normal (satu baris AWS2).")

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
    hist_files = st.file_uploader("Upload Excel History (format SCHDatabase)", type=["xlsx", "xlsm"],
                                  accept_multiple_files=True, key="hist_files",
                                  help="Riwayat jadwal tes yang disusun manual, dipakai HANYA untuk "
                                       "perbandingan rute di Tab Komparasi. File ini tidak disimpan ke "
                                       "SCH_Database dan tidak ikut menentukan COMP/NCMP/PENDING maupun "
                                       "kelayakan penjadwalan — untuk itu pakai menu Status Realisasi Harian. "
                                       "Kolom yang dibaca: WELL, UNIT, SCHEDULE_DATE_TEST.")

    st.divider()
    ui.section("🛣️ Optimasi Jarak Jalan Nyata (OSRM)")
    st.caption("Default optimasi memakai jarak garis lurus (haversine). Aktifkan ini agar "
               "jarak antar sumur memakai jaringan jalan nyata dari server OSRM. Pasangan yang "
               "belum ada di cache tetap jatuh ke haversine, jadi optimasi tidak pernah patah.")
    osrm_url = st.text_input("OSRM Base URL", "https://router.project-osrm.org",
                             help="Endpoint OSRM /table. Bisa server publik demo atau OSRM yang di-host sendiri. "
                                  "Server demo publik sering lambat & membatasi jumlah titik; untuk data besar "
                                  "disarankan host OSRM sendiri.")
    osrm_profile = st.text_input("Profil rute OSRM", "driving",
                                 help="Profil 'driving' = jalan yang bisa dilalui mobil (buang jalur kaki/motor). "
                                      "Ini profil mobil generik, BUKAN truk: tidak memperhitungkan berat/lebar/tinggi "
                                      "unit MWT, dan bergantung pada kelengkapan data OpenStreetMap. Untuk rute yang "
                                      "benar-benar sesuai unit MWT, host OSRM sendiri dengan profil kustom (mis. "
                                      "truck) lalu isi namanya di sini. Server demo publik hanya punya 'driving'.")
    _OSRM_INSECURE = st.checkbox("Abaikan verifikasi sertifikat SSL (OSRM)", value=False,
                                 help="Centang bila muncul error 'CERTIFICATE_VERIFY_FAILED' saat mengambil OSRM. "
                                      "Biasa terjadi di jaringan kantor yang memakai proxy penyaring (SSL inspection) "
                                      "dengan sertifikat sendiri. Hanya memengaruhi panggilan ke server OSRM.")
    _oc1, _oc2 = st.columns(2)
    with _oc1:
        osrm_chunk = st.number_input("Titik per permintaan", 5, 100, 40, step=5,
                                     help="Makin kecil makin ringan untuk server publik yang lambat, tapi butuh "
                                          "lebih banyak permintaan.")
    with _oc2:
        osrm_timeout = st.number_input("Timeout (detik)", 15, 600, 120, step=15,
                                       help="Perbesar bila server publik lambat merespons.")
    osrm_src = st.radio("Sumber koordinat", ["Sumur kandidat (dari Excel)", "Semua sumur Data Spasial"],
                        index=0, horizontal=True,
                        help="Sumur kandidat = hanya koordinat sumur yang ada di Excel kandidat (jauh lebih "
                             "sedikit titik, OSRM lebih ringan). Cache dipakai lintas periode, jadi cukup "
                             "sekali bangun untuk semua sumur kandidat.")
    if osrm_src.startswith("Sumur kandidat") and not spatial_db.empty:
        _cand_names = set(raw["well"].astype(str))
        _src_df = spatial_db[spatial_db["WELL"].astype(str).isin(_cand_names)]
    else:
        _src_df = spatial_db
    st.caption(f"🎯 {0 if _src_df is None or _src_df.empty else len(_src_df):,} sumur jadi sumber koordinat.")
    _road_n = _road_dist_count()   # jumlah untuk tampilan; TANPA memuat dict (hemat memori)
    _pq_n = "ada" if os.path.exists(ROAD_PARQUET) else "tidak ada"
    st.caption(f"📦 Cache jarak jalan: **{_road_n:,} pasangan** (basis Parquet repo: {_pq_n} + tambahan SQLite). "
               "Dimuat per periode & disaring ke sumur terkait saja agar hemat memori (anti-OOM).")
    if st.button("🔄 Bangun / Perbarui matriks jarak jalan", use_container_width=True,
                 help="Ambil jarak jalan untuk sumber koordinat terpilih via OSRM, lalu simpan ke cache lokal. "
                      "Hasil parsial tetap tersimpan bila server terputus di tengah jalan."):
        _coords = list(zip(_src_df["LAT"].astype(float), _src_df["LON"].astype(float))) \
                  if _src_df is not None and not _src_df.empty else []
        _uniq = list(dict.fromkeys(_rk(a, b) for a, b in _coords))
        if len(_uniq) < 2:
            st.warning("Butuh minimal 2 koordinat sumur pada sumber terpilih.")
        else:
            _pb = st.progress(0.0, text=f"Meminta OSRM untuk {len(_uniq)} titik…")
            try:
                n_pair, n_null = osrm_build_matrix(_coords, osrm_url,
                                                   chunk=int(osrm_chunk), timeout=int(osrm_timeout),
                                                   profile=(osrm_profile.strip() or "driving"),
                                                   progress=lambda f: _pb.progress(min(1.0, f)))
                _pb.empty()
                _route_cache_store().clear()
                load_road_dist_cached.clear()
                _road_n = _road_dist_count()
                msg = f"✅ {n_pair:,} pasangan jarak jalan tersimpan."
                if n_null:
                    msg += f" ({n_null:,} pasangan tak terjangkau, pakai haversine.)"
                st.success(msg)
            except Exception as e:
                _pb.empty()
                _route_cache_store().clear()
                load_road_dist_cached.clear()
                _road_n = _road_dist_count()
                _saved = _road_n
                st.error(f"Gagal mengambil OSRM: {e}")
                if _saved:
                    st.info(f"📦 {_saved:,} pasangan yang sempat terambil sudah tersimpan di cache. "
                            "Klik tombol lagi untuk melanjutkan sisanya (kecilkan 'Titik per permintaan' "
                            "atau perbesar 'Timeout' bila masih gagal).")
    _USE_ROAD = st.checkbox("Pakai jarak jalan nyata (OSRM) untuk optimasi", value=True,
                            disabled=(_road_n == 0),
                            help="Bila cache kosong, bangun matriks dulu. Saat aktif, jarak antar sumur "
                                 "yang ada di cache memakai jaringan jalan; sisanya haversine.")
    # _ROAD_KM/_ROAD_DETOUR diisi SETELAH filter periode (disaring ke sumur periode terpilih
    # saja) supaya dict tak menahan jutaan pasangan di memori. Di sini hanya placeholder.
    _ROAD_KM = {}
    _ROAD_DETOUR = 1.0
    if _USE_ROAD:
        st.caption("🛣️ Mode jarak jalan **aktif** — jarak jalan diterapkan untuk sumur periode terpilih "
                   "(cache disaring per periode agar hemat memori).")

    _DRAW_ROAD = st.checkbox("Gambar rute jalan nyata di peta (bukan garis lurus)", value=True,
                             help="Menarik geometri jalan dari OSRM /route saat peta dirender (sekali, lalu "
                                  "di-cache lokal). Pasangan yang belum ada geometrinya digambar garis lurus. "
                                  "Butuh koneksi ke server OSRM saat pertama kali render.")
    if _DRAW_ROAD:
        _ROAD_GEOM = load_road_geom()
        _GEOM_FAIL = False
        st.caption(f"🧭 Geometri jalan tersimpan: **{len(_ROAD_GEOM):,} pasangan**.")
    else:
        _ROAD_GEOM = {}
    _osrm_url_cfg, _osrm_to_cfg = osrm_url, int(osrm_timeout)
    _osrm_profile_cfg = osrm_profile.strip() or "driving"

    with st.expander("💾 Ekspor / Impor cache jarak jalan (file lokal portabel)"):
        st.caption("Simpan cache ke file agar tetap awet bila DB dihapus/pindah komputer, "
                   "atau bagikan ke pengguna lain tanpa perlu menarik ulang dari OSRM.")
        st.caption("Basis lengkap cache ada di file **road_dist_cache.parquet** (bawaan repo). "
                   "Ekspor di bawah hanya berisi **tambahan runtime di SQLite** (pasangan baru hasil "
                   "prefetch), yang belum masuk Parquet — gabungkan lalu ekspor ulang ke Parquet bila perlu.")
        _con_x = db_connect()
        try:
            _sdf = pd.read_sql("SELECT alat,alon,blat,blon,km FROM road_dist_cache", _con_x)
        except Exception:
            _sdf = pd.DataFrame(columns=["alat", "alon", "blat", "blon", "km"])
        _con_x.close()
        if len(_sdf):
            st.download_button(f"⬇️ Ekspor tambahan SQLite ({len(_sdf):,} pasangan, CSV)",
                               _sdf.to_csv(index=False).encode(),
                               "road_dist_cache_sqlite.csv", "text/csv", use_container_width=True)
        else:
            st.caption("Belum ada tambahan runtime di SQLite.")
        _imp = st.file_uploader("Impor cache (CSV hasil ekspor)", type=["csv"], key="road_imp")
        if _imp is not None:
            _sig = (getattr(_imp, "name", ""), getattr(_imp, "size", 0))
            if st.session_state.get("_road_imp_sig") != _sig:
                try:
                    _di = pd.read_csv(_imp)
                    _need = {"alat", "alon", "blat", "blon", "km"}
                    if not _need.issubset(_di.columns):
                        st.error("Kolom wajib: alat, alon, blat, blon, km.")
                    else:
                        _pairs = [(_rk(r.alat, r.alon), _rk(r.blat, r.blon), float(r.km))
                                  for r in _di.itertuples()
                                  if pd.notna(r.alat) and pd.notna(r.blat) and pd.notna(r.km)]
                        save_road_dist(_pairs)
                        st.session_state["_road_imp_sig"] = _sig
                        _route_cache_store().clear()
                        load_road_dist_cached.clear()
                        st.success(f"✅ {len(_pairs):,} pasangan diimpor ke cache.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Gagal impor: {e}")

    st.divider()

    ui.section("📅 Horizon Perencanaan")
    _today = datetime.now().date()
    sel_periode = None
    if HAS_PERIODS and period_opts is not None and len(period_opts):
        sel_periode = st.selectbox("Rentang Periode (Compiled Schedule)", list(period_opts.index),
                                   index=len(period_opts) - 1,
                                   help="WELLGO hanya memproses baris sumur pada rentang periode ini. "
                                        "Default menampilkan periode paling akhir (terkini).")
        per_lo = period_opts.loc[sel_periode, "_s"].date()
        _pe = period_opts.loc[sel_periode, "_e"]
        per_hi = _pe.date() if pd.notna(_pe) else (per_lo + timedelta(days=9))
        st.caption(f"📆 **{per_lo:%d %b %Y} – {per_hi:%d %b %Y}** · hanya baris periode ini yang diproses.")
    else:
        periode = st.date_input("Rentang Siklus (Periode)", value=(_today, _today + timedelta(days=6)),
                                help="Batas siklus keseluruhan. Menentukan data NCMP yang dibaca & kelayakan sumur.")
        if isinstance(periode, (list, tuple)) and len(periode) == 2:
            per_lo, per_hi = periode[0], periode[1]
        else:
            per_lo = periode if not isinstance(periode, (list, tuple)) else periode[0]
            per_hi = per_lo + timedelta(days=6)

    plan_start_date = st.date_input("Mulai Planning dari Tanggal",
                                    value=per_lo, min_value=per_lo, max_value=per_hi,
                                    help="Titik mulai optimasi rute. Penomoran hari tetap dihitung dari awal periode "
                                         "(mis. mulai hari ke-2 → dijadwalkan sebagai Hari 2).")

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
        mode_label = st.radio("Mode Distribusi Unit",
            ["Pooled (Bebas Zona)", f"Mapping Unit (sheet {SHEET_MAPUNIT})"], index=1,
            help="• Pooled: unit bebas, dibatasi zona remote/non-remote saja. "
                 f"• Mapping Unit: unit dibatasi daftar di sheet {SHEET_MAPUNIT} (SUB_AREA · FIELD · UNIT) "
                 "di file Excel kandidat yang sama. Satu lapangan boleh punya beberapa unit "
                 "(mis. 'MP445, MP523'); sumurnya tetap berkompetisi biasa, yang dibatasi hanya "
                 "unit mana yang berhak. Lapangan yang tidak ada di sheet tetap bebas seperti Pooled.")
        mode = "pooled"
        use_unitmap = mode_label.startswith("Mapping")
        crit_label = st.radio("Kriteria Utama Optimasi", list(CRIT.keys()), index=3)
        use_urg, use_dur = CRIT[crit_label]

        st.divider()
        max_wells = st.slider("Target Sumur / Unit / Hari", 3, 8, 6)
        min_wells = st.slider(
            "Minimum Sumur / Trip", 1, 4, 1,
            help="Cegah unit berangkat seharian cuma untuk 1–2 sumur. Kalau klaster sebuah unit "
                 "tak mencapai angka ini, WELLGO coba dulu mengisinya dgn radius 1,8×; kalau tetap "
                 "tipis, trip dibatalkan dan unitnya dikembalikan untuk lapangan lain. "
                 "Sumur mendesak (deadline hari itu/terlewat, NW/AWS, PRQ/ORQ) DIKECUALIKAN — "
                 "trip 1 sumur tetap jalan kalau memang wajib. Set 1 = perilaku lama.")
        n_remote = st.slider("Unit Area Remote (Bangko/Balam)", 1, 5, 5)
        n_nonremote = st.slider("Unit Area Non-Remote (Bekasap)", 1, 4, 4)
        
        elastic_limit = st.slider("Batas Persebaran Rute (Elastic Limit km)", 5, 50, 5, 1, help="Mencegah efek 'chaining' di mana armada merangkai jarak dekat tapi ujung ke ujungnya terlalu jauh.")

        regroup_prox = st.checkbox("Grouping ulang per hari berdasarkan kedekatan sumur", value=True,
            help="Berlaku untuk mode Pooled & Mapping Unit. Setelah aturan deadline menentukan sumur "
                 "mana digarap hari apa dan pakai berapa unit, keanggotaan tiap klaster disusun ULANG "
                 "murni dari kedekatan geografis. Hari, jumlah unit, dan daftar sumur terjadwal tidak "
                 "berubah sedikit pun — yang berubah hanya siapa berangkat bersama siapa, sehingga "
                 "rutenya lebih rapat. forced_unit (GP) & Mapping Unit tetap dihormati, dan klaster "
                 "yang tak bisa disusun ulang tanpa melanggar batasan dibiarkan apa adanya.")

        two_layer = False

        # ── Latihan skenario: batasi penjadwalan ke rentang deadline tertentu ──
        dl_scope = st.selectbox(
            "🎯 Cakupan Jadwal (latihan skenario)",
            ["Semua kandidat", "Hanya deadline di rentang", "Deadline di rentang + lainnya"],
            help="Latihan menjadwalkan dgn fokus pada hari-hari deadline tertentu. "
                 "• Hanya deadline di rentang: sumur di luar rentang TIDAK ikut dijadwalkan "
                 "sama sekali — dipakai melihat 'berapa kru yang dibutuhkan untuk menutup "
                 "deadline hari-hari ini saja'. "
                 "• Deadline di rentang + lainnya: sumur di rentang dioptimasi DULU (lapis 1), "
                 "sisa kapasitas baru diisi kandidat lain — cakupan penuh, prioritas jelas.")
        if dl_scope != "Semua kandidat":
            _rng = st.date_input("Rentang deadline yang difokuskan", (per_lo, per_hi),
                                 min_value=per_lo, max_value=per_hi, key="dl_scope_rng")
            if isinstance(_rng, (list, tuple)) and len(_rng) == 2:
                dl_rng_lo, dl_rng_hi = pd.Timestamp(_rng[0]), pd.Timestamp(_rng[1])
            else:
                _one = _rng[0] if isinstance(_rng, (list, tuple)) else _rng
                dl_rng_lo = dl_rng_hi = pd.Timestamp(_one)
        else:
            dl_rng_lo, dl_rng_hi = pd.Timestamp(per_lo), pd.Timestamp(per_hi)

        dl_mode = st.selectbox(
            "🎯 Jaminan Deadline", ["Off", "Sedang", "Agresif"], index=0,
            help="Sumur yang deadline-nya jatuh DI DALAM periode dikumpulkan lebih dulu di lapis 1 "
                 "dengan batas persebaran rute yang dilonggarkan. Tujuannya bukan menaikkan "
                 "prioritas (skoring urgensi sudah melakukannya), tapi MEMADATKAN sumur deadline "
                 "yang berjauhan ke sedikit unit — kalau tidak, 5 sumur berjauhan bisa memakan 3 unit "
                 "dan lapangan lain kehabisan kru. Sedang = 1,5× batas persebaran, Agresif = 2,5×. "
                 "Harganya: rute unit deadline jadi lebih panjang.")
        _DL_MULT = {"Off": 1.0, "Sedang": 1.5, "Agresif": 2.5}
        
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
# Penomoran hari relatif ke AWAL periode: mulai planning di hari ke-N → dijadwalkan sbg "Hari N".
day_offset = max(0, (plan_start_ts - per_lo_ts).days)

# Snapshot SEBELUM difilter periode: tab Analisis Performa perlu seluruh periode di Excel,
# sedangkan `raw` di bawah ini menyusut jadi satu periode terpilih saja.
raw_all_periods = raw.copy()

# Compiled Schedule: proses HANYA baris pada rentang periode terpilih (break-in selalu diikutkan).
if HAS_PERIODS and sel_periode is not None and "_rperiode" in raw.columns:
    raw = raw[(raw["_rperiode"] == sel_periode) | raw["is_breakin"].fillna(False)].copy()

# Cakupan overlay Status_Sumur utk periode terpilih — kalau 0, kunci join meleset
# (label Rentang Periode atau well_name beda), bukan sheet-nya yang kosong.
if "status_src" in raw.columns:
    _nov = int((raw["status_src"] == SHEET_STATUS).sum())
    if _nov:
        _noff = int((raw["status"] == "OFF").sum()) if "status" in raw.columns else 0
        st.sidebar.caption(f"🔄 {SHEET_STATUS}: {_nov}/{len(raw)} sumur ter-update · {_noff} OFF")
    else:
        st.sidebar.caption(f"⚠️ {SHEET_STATUS}: 0 sumur cocok — cek label Rentang Periode / well_name")

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

# ── Mode Mapping Unit ──────────────────────────────────────────────────────
# Dipasang SETELAH resolve_coords karena lapangan sumur bisa baru terisi dari
# master spasial di sana, dan pemetaan ini dikunci per (SUB_AREA, FIELD).
unit_map = load_unit_map(up.getvalue()) if use_unitmap else {}
raw["allow_units"] = unit_map_allow(raw, unit_map) if unit_map else None
unitmap_unknown = sorted({u for us in unit_map.values() for u in us} - set(ALL_UNITS))
if use_unitmap and not unit_map:
    st.sidebar.error(f"⚠️ Sheet **{SHEET_MAPUNIT}** tak terbaca (butuh kolom FIELD & UNIT). "
                     "Penjadwalan jalan seperti mode Pooled.")
elif unit_map:
    _n_cov = int(raw["allow_units"].notna().sum())
    st.sidebar.caption(f"🗺️ **Mapping Unit aktif** — {len(unit_map)} baris peta, "
                       f"{_n_cov}/{len(raw)} sumur terikat daftar unit."
                       + (f" ⚠️ unit tak dikenal: {', '.join(unitmap_unknown)}" if unitmap_unknown else ""))

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
# Nomor hari yang dipakai UI & day_idx — relatif AWAL periode, bukan 1..horizon.
# days[k] ⇄ day_nums[k]; konversi baliknya: days[day_idx - 1 - day_offset].
day_nums = list(range(1 + day_offset, horizon + 1 + day_offset))

executed_log, ncmp_log, pending_log = status_in_period(per_lo, per_hi)
comp_col = set(raw.loc[raw["sch_status"] == "COMP", "well"])
manual_comp = set(st.session_state.get("manual_comp", []))  # ditandai COMP manual oleh user (review SCH)

# ── AWS dua-fase (Option C): auto-transisi AWS1→AWS2 dari POP_Date + riwayat COMP ──
# AWS1 = POP+1..+3. Deteksi fase (COMP AWS1 → naik AWS2) tetap via POP/riwayat.
# CATATAN: window AWS2 final dihitung ulang dari last_wt_date (test date AWS1) di blok
# "Window fase lanjutan" setelah blok ini; POP+5..+10 hanya fallback saat last_wt kosong.
# Sumur baru dianggap "selesai" (executed) bila AWS2 sudah COMP.
aws_done = set()        # AWS2 selesai → final (executed)
aws_active = {}         # well → (fase, min_date|None, max_date|None); None = pertahankan window Excel
aws_extra_rows = []     # duplikat tugas AWS2 (mode "Pecah AWS jadi 2 kunjungan")
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
        ds = [pd.Timestamp(d).normalize() for d, _ in recs if pd.notna(d)]
        # Bukti fase HANYA dari SIKLUS BERJALAN. AWS1/AWS2 adalah tes setelah sumur
        # di-POP untuk siklus ini, jadi COMP yang mendahului POP milik siklus lampau dan
        # tak boleh menaikkan fase. Tanpa gerbang ini, sumur AWS1 dgn POP 30 Agu tapi
        # punya COMP 5 Agu ikut terhitung: n_comp>=1 menaikkannya ke AWS2 (window melompat
        # ke last_wt+5..+10 yang sudah lewat) atau n_comp>=2 menandainya selesai — kewajiban
        # AWS1 yang deadline-nya masih di periode ini LENYAP tanpa jejak, bahkan tak
        # muncul sbg Miss Deadline. Inilah yang membuat AWS1 & AWS2 seolah satu fase.
        _cyc = (pd.Timestamp(pop).normalize() if has_pop
                else (pd.Timestamp(_w["min_date"]).normalize() if pd.notna(_w.get("min_date")) else None))
        if _cyc is not None:
            ds = [d for d in ds if d >= _cyc]
        a1_win = a2_win = False
        if has_pop:
            popn = pd.Timestamp(pop).normalize()
            a1_win = any(popn + pd.Timedelta(days=1) <= d <= popn + pd.Timedelta(days=3) for d in ds)
            a2_win = any(popn + pd.Timedelta(days=5) <= d <= popn + pd.Timedelta(days=10) for d in ds)
        n_comp = len(ds)
        # ── Window AWS2 = tanggal AWS1 BENAR-BENAR dites + 5..+10, BUKAN dari POP ──
        # Sumber tanggal AWS1: kolom last_wt_date (tanggal tes fase sebelumnya) bila ada,
        # kalau tidak, COMP AWS1 terbaru di siklus ini. Selama AWS1 belum ada bukti dites,
        # window Excel dipertahankan (jangan mengarang dari POP). last_wt yang ada nanti
        # tetap ditegaskan lagi oleh blok "Window fase lanjutan" di bawah.
        _aws1_test = (pd.Timestamp(_w["last_wt"]).normalize() if pd.notna(_w.get("last_wt"))
                      else (max(ds) if ds else None))
        a2_ovr = ((_aws1_test + pd.Timedelta(days=5), _aws1_test + pd.Timedelta(days=10))
                  if (not excel_aws2 and _aws1_test is not None) else (None, None))
        # MODE PECAH 2 KUNJUNGAN: AWS1 belum ada bukti selesai & kedua window masuk periode terpilih
        # → base jadi tugas AWS1 (POP+1..+3), plus duplikat tugas AWS2. Di preview ini AWS1 belum
        # dites, jadi window AWS2 ditaksir relatif window AWS1: [AWS1_lo+5, AWS1_hi+10].
        if aws_split and has_pop and not excel_aws2 and not (as1 or as2) and n_comp == 0:
            a1_lo, a1_hi = popn + pd.Timedelta(days=1), popn + pd.Timedelta(days=3)
            # AWS2 relatif window AWS1 (bukan POP langsung): dari AWS1 paling awal +5
            # sampai AWS1 paling akhir +10, mencakup semua kemungkinan tanggal tes AWS1.
            a2_lo, a2_hi = a1_lo + pd.Timedelta(days=5), a1_hi + pd.Timedelta(days=10)
            a1_in = (a1_lo <= per_hi_ts) and (a1_hi >= per_lo_ts)
            a2_in = (a2_lo <= per_hi_ts) and (a2_hi >= per_lo_ts)
            if a1_in and a2_in:
                aws_active[_wn] = ("AWS1", a1_lo, a1_hi)     # base = AWS1
                _r2 = _w.to_dict()
                _r2["well"] = f"{_wn} #AWS2"
                _r2["category"] = "AWS2"; _r2["min_date"] = a2_lo; _r2["max_date"] = a2_hi
                aws_extra_rows.append(_r2)
                continue
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

    if aws_extra_rows:
        _dtc = [c for c in raw.columns if pd.api.types.is_datetime64_any_dtype(raw[c])]
        raw = pd.concat([raw, pd.DataFrame(aws_extra_rows)], ignore_index=True)
        for _c in _dtc:  # concat bisa merusak dtype datetime → paksa balik
            raw[_c] = pd.to_datetime(raw[_c], errors="coerce")

# ── Window fase lanjutan dari kolom last_wt_date (tanggal test fase SEBELUMNYA) ──
# AWS2       : min = last_wt + 5,  max = last_wt + 10   (last_wt = test date AWS1)
# NEW WELL 2 : min = last_wt + 10, max = last_wt + 15   (last_wt = test date NW1)
# NEW WELL 3 : min = last_wt + 10, max = last_wt + 15   (last_wt = test date NW2)
# last_wt kosong → window dari Excel dipertahankan (fallback). AWS2 di sini menimpa window POP lama.
if "last_wt" in raw.columns:
    _lw = raw["last_wt"]
    _catu = raw["category"].astype(str).str.upper().str.strip()
    _phase_rules = [
        (_catu.eq("AWS2"),                                            5, 10),
        (_catu.str.contains(r"NEW WELL\s*2\b", regex=True, na=False), 10, 15),
        (_catu.str.contains(r"NEW WELL\s*3\b", regex=True, na=False), 10, 15),
    ]
    for _mask, _lo, _hi in _phase_rules:
        _m = _mask & _lw.notna()
        raw.loc[_m, "min_date"] = _lw[_m] + pd.Timedelta(days=_lo)
        raw.loc[_m, "max_date"] = _lw[_m] + pd.Timedelta(days=_hi)

# ── COMP hanya berlaku untuk WINDOW YANG BERJALAN ─────────────────────────
# Kolom SCH STATUS dan riwayat execution_log memuat status tes TERAKHIR tanpa
# batas periode, jadi sumur AWS yang COMP di Juli tetap tertandai COMP saat
# periode September — padahal window barunya menuntut tes ulang, dan sumurnya
# jadi tak pernah masuk kandidat. Bukti COMP baru dihitung bila tanggal tes
# terakhirnya >= min_date baris ini (pembukaan window sekarang). min_date di sini
# sudah final: blok "Window fase lanjutan" di atas sudah menghitung ulang dari
# last_wt. executed_log TIDAK ikut digerbang — ia memang sudah period-scoped;
# manual_comp juga tidak, karena itu penandaan sengaja oleh user.
_last_test = {}
for _w2, _rc2 in comp_records(set(raw["well"])).items():
    _dd2 = [pd.Timestamp(d) for d, _ in _rc2 if pd.notna(d)]
    if _dd2:
        _last_test[_w2] = max(_dd2)
# last_wt = tanggal tes TERAKHIR, yang bisa saja percobaan NCMP (gagal). Itu bukan
# bukti selesai, jadi baris ber-SCH STATUS NCMP tak boleh menyumbang tanggal.
# Riwayat execution_log di atas sudah aman: comp_records hanya memuat status='executed'.
if "last_wt" in raw.columns:
    _ncmp_row = (raw["sch_status"].astype(str).str.upper().str.strip().eq("NCMP")
                 if "sch_status" in raw.columns else pd.Series(False, index=raw.index))
    for _w2, _lw2, _bad in zip(raw["well"], raw["last_wt"], _ncmp_row):
        if pd.notna(_lw2) and not _bad:
            _p = pd.Timestamp(_lw2)
            _last_test[_w2] = max(_last_test[_w2], _p) if _w2 in _last_test else _p
_minmap = dict(zip(raw["well"], raw["min_date"]))

_ncmp_wells = (set(raw.loc[raw["sch_status"].astype(str).str.upper().str.strip().eq("NCMP"), "well"])
               if "sch_status" in raw.columns else set())

def comp_masih_berlaku(w):
    """True bila COMP-nya masih menghitung utk window sekarang.
    • SCH STATUS = NCMP → tegas TIDAK selesai. Ini bukan ketidaktahuan: percobaan
      terakhirnya gagal, jadi jangan jatuh ke fallback "pertahankan lama".
    • Tanpa bukti tanggal sama sekali → tak bisa dinilai, pertahankan perilaku lama."""
    if w in _ncmp_wells:
        return False
    md, lt = _minmap.get(w), _last_test.get(w)
    if lt is None or pd.isna(md):
        return True
    return lt >= md

_comp_col_raw, _aws_done_raw = set(comp_col), set(aws_done)
comp_col = {w for w in comp_col if comp_masih_berlaku(w)}
aws_done = {w for w in aws_done if comp_masih_berlaku(w)}
comp_expired = (_comp_col_raw - comp_col) | (_aws_done_raw - aws_done)

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

ncmp_col_df = pd.DataFrame({"well": sorted(ncmp_col), "reason": "", "comment": "", "plan_date": "(kolom)"})
ncmp_df = pd.concat([ncmp_log, ncmp_col_df], ignore_index=True).drop_duplicates("well")
if "comment" not in ncmp_df.columns:
    ncmp_df["comment"] = ""
ncmp_df["comment"] = ncmp_df["comment"].fillna("").astype(str)
ncmp_df["kode_hambatan"] = ncmp_df["comment"].map(carry_code)

# Muat cache jarak jalan HANYA untuk pasangan antar sumur periode ini (disaring), lalu isi
# _ROAD_KM/_ROAD_DETOUR. Dilakukan di sini (setelah filter periode) agar dict kecil dan tak
# menahan jutaan pasangan di memori — penyebab OOM "Oh no. Error running app." di Streamlit Cloud.
if _USE_ROAD:
    _need_pts = frozenset(
        _rk(la, lo) for la, lo in zip(
            pd.to_numeric(raw["lat"], errors="coerce").tolist(),
            pd.to_numeric(raw["lon"], errors="coerce").tolist())
        if pd.notna(la) and pd.notna(lo))
    if _need_pts:
        # hash sendiri (kecil & stabil) sbg kunci cache; frozenset besar diberikan lewat _need
        # yang tak ikut di-hash Streamlit, supaya tiap rerun tak menghash ribuan koordinat.
        _ROAD_KM, _ROAD_DETOUR = load_road_dist_cached(_road_dist_sig(), hash(_need_pts), _need=_need_pts)

_E = build_elig(raw, ncmp_df, per_lo_ts, per_hi_ts, week_lo, week_hi,
                executed, comp_disp_set, pending_set, ncmp_replan, woff_set)
batch_lo, batch_hi = _E["batch_lo"], _E["batch_hi"]
cand, comp_wells, elig, elig_all = _E["cand"], _E["comp_wells"], _E["elig"], _E["elig_all"]
expired_df, ncmp_carry, ncmp_expired = _E["expired_df"], _E["ncmp_carry"], _E["ncmp_expired"]
ncmp_replan, nocoord, off_wells = _E["ncmp_replan"], _E["nocoord"], _E["off_wells"]
pending_nodata, pending_wells = _E["pending_nodata"], _E["pending_wells"]
replan_df, woff_wells = _E["replan_df"], _E["woff_wells"]

# ── Unit MWT Tidak Tersedia per Tanggal (blackout) — input ada di sidebar ───
unit_blackout_by_day = {}
for (_u, _dk) in st.session_state.get("unit_blackout", []):
    unit_blackout_by_day.setdefault(_dk, set()).add(_u)

# ── Rollout Execution Framework ────────────────────────────────────────────
route_anchors = st.session_state.get("route_anchors", {}) or None

# ── Cakupan Jadwal: fokuskan ke sumur ber-deadline di rentang terpilih ─────
# "Hanya"  → sumur di luar rentang dibuang dari kandidat (latihan kapasitas murni).
# "+lainnya" → tetap semua kandidat, tapi yang di rentang naik ke lapis 1 (lihat
#              _prio_mask di bawah) sehingga dioptimasi lebih dulu.
# Add Manual dikecualikan dari prioritas: perannya filler.
if len(elig):
    _sc_am = elig["is_addmanual"].fillna(False) if "is_addmanual" in elig.columns else pd.Series(False, index=elig.index)
    dl_scope_win = (elig["max_date"].notna() & (elig["max_date"] >= dl_rng_lo)
                    & (elig["max_date"] <= dl_rng_hi) & ~_sc_am)
    if dl_scope == "Hanya deadline di rentang":
        _n_before = len(elig)
        elig = elig[dl_scope_win].copy()
        dl_scope_win = dl_scope_win.loc[elig.index]
        scope_note = (f"🎯 **Cakupan: hanya deadline {dl_rng_lo:%d %b}–{dl_rng_hi:%d %b}** — "
                      f"{len(elig)} dari {_n_before} kandidat ikut dijadwalkan, sisanya "
                      f"sengaja dikesampingkan untuk latihan ini.")
    elif dl_scope == "Deadline di rentang + lainnya":
        scope_note = (f"🎯 **Cakupan: deadline {dl_rng_lo:%d %b}–{dl_rng_hi:%d %b} didahulukan** — "
                      f"{int(dl_scope_win.sum())} sumur masuk lapis 1, "
                      f"{int((~dl_scope_win).sum())} kandidat lain mengisi sisa kapasitas.")
    else:
        scope_note = ""
else:
    dl_scope_win = pd.Series(dtype=bool); scope_note = ""

# ── Jaminan Deadline ───────────────────────────────────────────────────────
# Sumur yg deadline-nya (max_date) jatuh DI DALAM periode wajib kebagian kru sebelum
# lewat, tapi skoring lapangan berbasis kedekatan bikin mereka kalah dari lapangan
# padat yg deadline-nya masih jauh. Mode ini menaikkan mereka ke lapis 1.
# Add Manual dikecualikan: perannya filler, tak boleh menggeser sumur lain.
if len(elig):
    _dl_am = elig["is_addmanual"].fillna(False) if "is_addmanual" in elig.columns else pd.Series(False, index=elig.index)
    dl_guard = (elig["max_date"].notna() & (elig["max_date"] >= batch_lo)
                & (elig["max_date"] <= batch_hi) & ~_dl_am) if dl_mode != "Off" else pd.Series(False, index=elig.index)
else:
    dl_guard = pd.Series(dtype=bool)

if len(elig):
    if two_layer or dl_mode != "Off" or dl_scope == "Deadline di rentang + lainnya":
        # Lapis 1: sumur prioritas (NW/AWS/PRQ/ORQ + carry NCMP) dioptimasi lebih dulu
        _prio_mask = elig["is_nwaws"].fillna(False) | elig["req_tag"].isin(["PRQ", "ORQ"]) | elig["carry_ncmp"].fillna(False) | dl_guard
        if dl_scope == "Deadline di rentang + lainnya":
            _prio_mask = _prio_mask | dl_scope_win
        prio_elig = elig[_prio_mask].copy()
        reg_elig = elig[~_prio_mask].copy()
        # Longgarkan batas persebaran HANYA di lapis 1, supaya sumur deadline yang berjauhan
        # muat dalam satu klaster & tak memakan unit ekstra. Rute reguler tak ikut longgar.
        _el_prio = min(50.0, elastic_limit * _DL_MULT.get(dl_mode, 1.0))
        if len(prio_elig):
            wk_prio = plan_week(prio_elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, _el_prio, unit_blackout=unit_blackout_by_day, min_wells=min_wells, anchors=route_anchors, day_offset=day_offset)
        else:
            wk_prio = prio_elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
        # Lapis 2: reguler mengisi sisa kapasitas; sumur prioritas terjadwal jadi 'prebooked'
        pb = wk_prio[wk_prio["scheduled"]].copy()
        if len(reg_elig):
            wk_reg = plan_week(reg_elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, unit_blackout=unit_blackout_by_day, min_wells=min_wells, anchors=route_anchors, prebooked=pb if len(pb) else None, day_offset=day_offset)
        else:
            wk_reg = reg_elig.assign(scheduled=False, plan_unit=None, plan_day=pd.NaT, day_idx=0)
        week_df = pd.concat([wk_prio, wk_reg], ignore_index=True)
    else:
        week_df = plan_week(elig, days, mode, max_wells, n_remote, n_nonremote, time_budget, speed, use_urg, use_dur, early_days, elastic_limit, unit_blackout=unit_blackout_by_day, min_wells=min_wells, anchors=route_anchors, day_offset=day_offset)
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
        week_df.loc[m, "plan_day"] = days[max(0, min(len(days) - 1, di - 1 - day_offset))]  # di relatif periode
        week_df.loc[m, "manual"] = True

man_un = st.session_state.get("manual_unassign", [])
if man_un:
    m_un = week_df["well"].isin(man_un)
    week_df.loc[m_un, "scheduled"] = False
    week_df.loc[m_un, "plan_unit"] = None
    week_df.loc[m_un, "day_idx"] = 0
    week_df.loc[m_un, "plan_day"] = pd.NaT
    week_df.loc[m_un, "manual"] = False

# ── Grouping ulang murni kedekatan (Pooled & Mapping Unit) ─────────────────
# Dijalankan SETELAH seluruh penugasan final (termasuk assign manual) supaya yang
# disusun ulang benar-benar jadwal yang akan dipakai.
_regroup_info = None
if regroup_prox and mode == "pooled":
    week_df, _regroup_info = regroup_by_proximity(week_df, max_wells)

scheduled_all = week_df[week_df["scheduled"]].copy()

# Laporkan hasil grouping ulang: berapa yang berpindah unit & berapa km yang dihemat.
if _regroup_info and (_regroup_info["grup"] or _regroup_info.get("tak_untung")):
    _ri = _regroup_info
    _hemat = _ri["km_awal"] - _ri["km_akhir"]        # dijamin ≥ 0 oleh gerbang anti-boros
    _pct = (100.0 * _hemat / _ri["km_awal"]) if _ri["km_awal"] > 0 else 0.0
    if _ri["grup"]:
        st.caption(f"🧲 **Grouping ulang kedekatan** — {_ri['grup']} klaster (unit×hari) disusun ulang, "
                   f"{_ri['pindah']} sumur pindah unit. Total rute "
                   f"**{_ri['km_awal']:.1f} → {_ri['km_akhir']:.1f} km** (hemat {_hemat:.1f} km · {_pct:.1f}%). "
                   "Hari, jumlah unit, & daftar sumur terjadwal tidak berubah."
                   + (f" {_ri['tak_untung']} klaster dibiarkan karena penyusunan ulang malah menambah jarak."
                      if _ri.get("tak_untung") else "")
                   + (f" {_ri['gagal']} klaster dilewati karena batasan forced_unit/mapping."
                      if _ri["gagal"] else ""))
    else:
        st.caption(f"🧲 **Grouping ulang kedekatan dilewati** — {_ri['tak_untung']} klaster diperiksa, "
                   "tak satu pun lebih hemat jaraknya, jadi susunan asli dipertahankan.")

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
# Sumur tak terjadwal dgn deadline <= akhir periode. Pisahkan lagi:
#   • missed         → deadline BENAR-BENAR di dalam periode  = Miss Deadline asli (miss kapasitas)
#   • missed_outside → deadline sudah lewat SEBELUM periode mulai (window min-max di luar periode)
#   • missed_carry   → NCMP FACI/ROAD/WOFF yang tak kebagian slot sampai akhir periode
# Yang di luar jangan dihitung sbg Miss Deadline: bukan gagal kebagian kru, tapi memang lewat window.
# NCMP FACI/ROAD/WOFF disaring LEBIH DULU: sampai akhir periode tak terjadwal pun ia
# BUKAN Miss Deadline (hambatan lapangan, bukan kuota kru), melainkan kategori sendiri
# "Not Complete (NCMP) FACI/ROAD/WOFF-Miss Deadline".
if len(leftover):
    # Gerbang deadline sama dgn missed_all: carry yang deadline-nya masih SETELAH periode
    # belum gagal apa-apa — biarkan jadi "ditunda" seperti kandidat lain.
    _is_carry = leftover["well"].isin(ncmp_carry) & (leftover["max_date"] <= batch_hi)
    missed_carry = leftover[_is_carry].copy()
    missed_carry["kode_hambatan"] = missed_carry["well"].map(ncmp_carry)
    missed_carry["kategori_ncmp"] = missed_carry["kode_hambatan"].map(carry_label)
    leftover_dl = leftover[~_is_carry]
else:
    missed_carry = leftover.iloc[0:0].copy()
    missed_carry["kode_hambatan"] = missed_carry["kategori_ncmp"] = None
    leftover_dl = leftover

missed_all = leftover_dl[leftover_dl["max_date"] <= batch_hi] if len(leftover_dl) else leftover_dl.copy()
if len(missed_all):
    _dl_in_period = missed_all["max_date"] >= batch_lo
    missed = missed_all[_dl_in_period].copy()
    missed_outside = missed_all[~_dl_in_period].copy()
else:
    missed = missed_all.copy()
    missed_outside = leftover_dl.iloc[0:0].copy()

# ── Render Header & KPIs via WELLGO UI ──────────────────────────────────────
total_scheduled = len(scheduled_all)
total_missed_dl = len(missed)
total_carry_miss = len(missed_carry)
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

# Gas well (GP) terjadwal + totalnya di pool kandidat sbg pembanding
_gp_s = (scheduled_all["is_gp"].fillna(False)
         if ("is_gp" in scheduled_all.columns and len(scheduled_all)) else pd.Series(dtype=bool))
n_gp = int(_gp_s.sum())
n_gp_cand = int(raw["is_gp"].fillna(False).sum()) if "is_gp" in raw.columns else 0
_gp_al = _gp_s.reindex(_dur_sched.index, fill_value=False) if n_gp else _dur_sched.astype(bool) & False
n_gp60 = int(((_dur_sched == 60) & _gp_al).sum())
n_gp30 = int(((_dur_sched == 30) & _gp_al).sum())

ui.hero_header(
    date_str=plan_start_ts.strftime("%d %b %Y"), 
    horizon=horizon, 
    units=len(scheduled_all["plan_unit"].unique()) if len(scheduled_all) else 0, 
    compliance=comp_rate, 
    mode=("mapping unit" if unit_map else mode)
)

# Miss deadline dipecah 2 kartu supaya sebabnya kebaca langsung dari header:
#   Pure  = gagal kebagian kru/kapasitas  → aksi: tambah shift/unit.
#   NCMP  = terhalang FACI/ROAD/WOFF      → aksi: benahi fasilitas/akses/status sumur.
ui.kpi_row([
    ("wells scheduled",     f"{total_scheduled}", f"/{total_eligible}", ui.TEAL_GREEN),
    ("pure-miss deadline",  f"{total_missed_dl}", " wells", ui.RED),
    ("ncmp-miss deadline",  f"{total_carry_miss}", " wells", "#E67E22"),
    ("wells off",           f"{len(off_wells)}", " wells", "#64748B"),
    ("total route",         f"{computed_total_km:.0f}", " km", ui.TEAL),
    ("avg utilization",     f"{avg_utilization:.0f}", "%",  ui.AMBER),
])
st.caption("🎯 **Pure-Miss Deadline** = deadline jatuh di dalam periode tapi kuota kru habis. "
           f"🚧 **NCMP-Miss Deadline** = *{NCMP_CARRY_LABEL}* — sudah dibawa ulang sepanjang "
           "periode karena hambatan lapangan (COMMENT IF NOT COMPLETE = FACI/ROAD/WOFF), tetap "
           "tak dapat slot. Dua-duanya dipisah karena tindak lanjutnya beda: tambah shift/unit "
           "vs benahi fasilitas, akses jalan, atau status sumur.")
_dur_caption = (f"🗓️ **{total_scheduled} sumur terjadwal** — "
                f"🕐 tes 60 menit: **{n_dur60}** · 🕧 tes 30 menit: **{n_dur30}**"
                + f" · ⛽ gas well (GP): **{n_gp}**"
                + (f"/{n_gp_cand} kandidat" if n_gp_cand else "")
                + (f" (60 mnt: {n_gp60} · 30 mnt: {n_gp30})" if n_gp else "")
                + (f" · durasi lain: **{n_dur_other}**" if n_dur_other else ""))
st.caption(_dur_caption)
if ncmp_carry:
    _cc = pd.Series(list(ncmp_carry.values())).value_counts()
    _sched_carry = len([w for w in ncmp_carry if w in sched_wells])
    st.caption(f"🚧 **{len(ncmp_carry)} NCMP hambatan lapangan** dibawa ulang periode ini — "
               + " · ".join(f"**{k}**: {int(v)}" for k, v in _cc.items())
               + f" · berhasil terjadwal: **{_sched_carry}**, sisanya jatuh ke kartu "
               "**ncmp-miss deadline** atau ditunda bila deadline-nya masih di luar periode.")
if scope_note:
    st.info(scope_note)

if comp_expired:
    st.info(f"♻️ **{len(comp_expired)} sumur dikembalikan jadi kandidat** — ditandai COMP oleh kolom "
            f"SCH STATUS / riwayat, tapi tes terakhirnya MENDAHULUI pembukaan window periode ini, "
            f"jadi harus dites lagi. Contoh: {', '.join(sorted(comp_expired)[:8])}"
            + (" …" if len(comp_expired) > 8 else ""))

st.caption(f"🔌 **WELLS OFF ({len(off_wells)})** = sumur OFF yang jadi kandidat & di-skip **di siklus ini**. "
           f"Total semua sumur OFF di master data: **{len(master_off_wells)}** — lihat daftar lengkapnya di tab "
           f"**⭐ Prioritas & Status Khusus**.")

# ── Rapor Jaminan Deadline ─────────────────────────────────────────────────
# Tanpa ini mode Jaminan Deadline tak terverifikasi: tepat waktu = terjadwal pada
# hari <= max_date. Terjadwal LEWAT deadline dihitung terlambat, bukan sukses.
if dl_mode != "Off" and bool(dl_guard.any()):
    _dl_wells = set(elig.loc[dl_guard, "well"])
    _ds = scheduled_all[scheduled_all["well"].isin(_dl_wells)] if len(scheduled_all) else scheduled_all
    _ontime = int((_ds["plan_day"] <= _ds["max_date"]).sum()) if len(_ds) else 0
    _late = len(_ds) - _ontime
    _unsched = len(_dl_wells) - len(_ds)
    _pct = int(100 * _ontime / len(_dl_wells)) if _dl_wells else 100
    _dl_units = int(_ds["plan_unit"].nunique()) if len(_ds) else 0
    _msg = (f"🎯 **Jaminan Deadline ({dl_mode}, {_DL_MULT[dl_mode]:g}× batas persebaran)** — "
            f"{len(_dl_wells)} sumur berdeadline dalam periode: **{_ontime} tepat waktu ({_pct}%)**, "
            f"dikerjakan **{_dl_units} unit**"
            + (f" · ⏰ {_late} terjadwal lewat deadline" if _late else "")
            + (f" · ❌ {_unsched} tak terjadwal" if _unsched else ""))
    (st.success if (_late + _unsched) == 0 else st.warning)(_msg)
    if _late + _unsched:
        with st.expander(f"⚠️ {_late + _unsched} sumur berdeadline belum aman"):
            _risk = elig[dl_guard & elig["well"].isin(
                (_dl_wells - set(_ds["well"])) | set(_ds.loc[_ds["plan_day"] > _ds["max_date"], "well"]
                                                    if len(_ds) else []))][
                ["well", "field", "area", "min_date", "max_date"]].copy()
            _risk["sisa hari"] = (_risk["max_date"] - week_lo).dt.days
            st.dataframe(_risk.sort_values("max_date"), use_container_width=True, hide_index=True)
            st.caption("Kalau masih banyak yang merah di mode **Ketat**, kapasitasnya yang kurang — "
                       "tambah unit/hari, naikkan Target Sumur/Unit/Hari, atau geser sebagian ke periode lain.")

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

(tab_guide, tab_sched, tab_map, tab_matrix, tab_cart, tab_sch, tab_diagnostics, tab_priority,
 tab_export, tab_compare, tab_bench, tab_candidates) = st.tabs([
    "📘 Panduan", "📅 Jadwal Operasional", "🗺️ Peta Rute", "📊 Matriks Deviasi", "🛒 Cart Manual",
    "🗃️ SCH Database", "📏 Analisis Jarak", "⭐ Prioritas & Status Khusus", "📤 Export", "⚖️ Komparasi",
    "🏁 Analisis Performa", "🧾 Kandidat"
])

with tab_bench:
    ui.section("Analisis Performa: Manual vs WELLGO", eyebrow="Seluruh periode di Excel · compliance & km/well")
    st.caption("Tiap periode di sheet kandidat dijadwalkan **ulang dari nol** dengan parameter algoritma "
               "yang sedang aktif di sidebar, sekali untuk mode **Pooled** dan sekali untuk **Mapping Unit**, "
               "lalu diukur compliance dan km/well-nya. Status realisasi (COMP/NCMP/PENDING) sengaja "
               "**tidak** dipakai di sini: kalau hasil periode itu diintip, sumur yang dulu sudah dites "
               "keluar dari kandidat dan WELLGO seolah tak punya pekerjaan. Sisi **Manual** diambil dari "
               "file history di sidebar — km/well saja, compliance-nya tak bisa dihitung karena file itu "
               "tak memuat deadline sumur.")

    if not (HAS_PERIODS and period_opts is not None and len(period_opts)):
        st.info("💡 Analisis lintas periode butuh sheet kandidat ber-kolom **Rentang Periode** beserta "
                "Start/End Periode (format Compiled Schedule). Sheet yang dipakai sekarang tidak punya "
                "kolom itu, jadi hanya ada satu periode untuk dianalisis.")
    else:
        _oil_col = find_oil_col(raw_all_periods)
        _bench_umap = unit_map or load_unit_map(up.getvalue())
        c_run, c_note = st.columns([1, 3])
        run_bench = c_run.button("🏁 Jalankan Analisis", type="primary", use_container_width=True,
                                 key="btn_bench")
        c_note.caption(f"**{len(period_opts)} periode** akan dijadwalkan ulang × 2 mode. "
                       + (f"Lost Oil dari kolom **{_oil_col}**."
                          if _oil_col else "⚠️ Kolom laju minyak tak ditemukan di sheet kandidat "
                          "(dicari nama yang memuat BOPD/NET OIL/OIL RATE/OIL) — kolom Lost Oil dikosongkan.")
                       + ("" if _bench_umap else " ⚠️ Sheet Mapping_Unit tak terbaca — kolom Mapping Unit "
                          "akan sama dengan Pooled."))
        # Unit yang dipetakan tapi di luar slider jumlah unit membuat kolom Mapping Unit anjlok
        # karena sumurnya tak punya unit yang berhak, bukan karena modenya buruk. Sebut di depan.
        _idle_b = sorted(({u for us in _bench_umap.values() for u in us} & set(ALL_UNITS))
                         - (set(REMOTE_UNITS[:n_remote]) | set(NONREMOTE_UNITS[:n_nonremote])))
        if _idle_b:
            st.warning(f"⚠️ Unit **{', '.join(_idle_b)}** dipetakan di sheet Mapping_Unit tapi tidak aktif "
                       "di slider jumlah unit. Sumur yang hanya boleh digarap unit ini tak akan pernah "
                       "terjadwal, sehingga kolom **Mapping Unit** akan terlihat jauh lebih buruk dari "
                       "Pooled bukan karena modenya, melainkan karena kapasitasnya dipotong. Naikkan "
                       "slider Unit Area Remote/Non-Remote lebih dulu agar perbandingannya adil.")

        if run_bench:
            _params = dict(max_wells=max_wells, n_remote=n_remote, n_nonremote=n_nonremote,
                           time_budget=time_budget, speed=speed, use_urg=use_urg, use_dur=use_dur,
                           early_days=early_days, elastic_limit=elastic_limit, min_wells=min_wells)
            _coord_cache = load_coord_cache()
            _hist = sch_history([f.getvalue() for f in hist_files]) if hist_files else pd.DataFrame()
            _cmap = {"lat": {}, "lon": {}}
            if not field_wells_coord.empty:
                _fw = field_wells_coord.drop_duplicates(subset=["well"]).set_index("well")
                _cmap = {"lat": _fw["lat"].to_dict(), "lon": _fw["lon"].to_dict()}

            rows = []
            miss_rows = []
            prog = st.progress(0.0, text="Menjadwalkan ulang tiap periode…")
            for _i, _pr in enumerate(period_opts.index, start=1):
                _lo = period_opts.loc[_pr, "_s"].date()
                _pe = period_opts.loc[_pr, "_e"]
                _hi = _pe.date() if pd.notna(_pe) else (_lo + timedelta(days=9))
                prog.progress(_i / len(period_opts), text=f"Periode {_pr} ({_i}/{len(period_opts)})…")

                # Siapkan kandidat periode ini dgn urutan filter yang sama seperti run utama.
                _r = raw_all_periods[(raw_all_periods["_rperiode"] == _pr)
                                     | raw_all_periods["is_breakin"].fillna(False)].copy()
                if excl_areas: _r = _r[~_r["area"].isin(excl_areas)].copy()
                if mpas_only:
                    _pl = (((_r["is_mpas"] | _r["unit_unknown"]) & ~_r["area"].isin(mwt_unavail))
                           | (_r["is_ts"] & _r["area"].isin(ts_unavail)))
                    _r = _r[_pl].copy()
                    _r.loc[_r["is_ts"] & _r["area"].isin(ts_unavail), "dur"] = 60
                if not len(_r):
                    continue
                _r = resolve_coords(_r, spatial_db, _coord_cache, field_assign=field_assign)

                _pool = bench_period(_r, _lo, _hi, {}, _params, _oil_col)
                if _pool is None:
                    continue
                _mapu = bench_period(_r, _lo, _hi, _bench_umap, _params, _oil_col)
                # Deadline sisi Manual diambil dari pool kandidat periode ini (kolom min–max
                # sheet kandidat), bukan dari file history yang memang tak memuatnya.
                _man = bench_manual(_hist, _lo, _hi, _cmap, _pool["pool"])
                rows.append({
                    "Periode": _pr, "Mulai": _lo, "Selesai": _hi, "Kandidat": _pool["kandidat"],
                    "Manual · Compliance, %": (_man or {}).get("compliance", np.nan),
                    "Manual · km/well": (_man or {}).get("km_well", np.nan),
                    "Pooled · Compliance, %": _pool["compliance"],
                    "Pooled · km/well": _pool["km_well"],
                    "Mapping Unit · Compliance, %": (_mapu or _pool)["compliance"],
                    "Mapping Unit · km/well": (_mapu or _pool)["km_well"],
                    "Lost Oil (delta)": ((_mapu or _pool)["lost_oil"] - _pool["lost_oil"]
                                         if _oil_col else np.nan),
                })
                # Kumpulkan sumur Not-Comply (miss deadline) per periode & per mode.
                # Manual ikut bila ada file history (baru saat itu compliance manual bermakna).
                for _mode_name, _res in (("Manual", _man),
                                         ("Pooled (bebas zona)", _pool),
                                         ("Mapping Unit", _mapu or _pool)):
                    _md = (_res or {}).get("miss_df")
                    if _md is not None and len(_md):
                        _t = _md.copy()
                        _t.insert(0, "Mode", _mode_name)
                        _t.insert(0, "Periode", _pr)
                        miss_rows.append(_t)
            prog.empty()
            st.session_state["_bench_rows"] = rows
            st.session_state["_bench_miss"] = (pd.concat(miss_rows, ignore_index=True)
                                               if miss_rows else pd.DataFrame())
            st.session_state["_bench_oil"] = _oil_col

        rows = st.session_state.get("_bench_rows")
        if not rows:
            st.info("Klik **Jalankan Analisis** untuk menghitung. Hasilnya tersimpan sampai Anda "
                    "menjalankannya lagi, jadi mengganti tab tidak menghitung ulang.")
        else:
            bdf = pd.DataFrame(rows)
            _num = [c for c in bdf.columns if "Compliance" in c or "km/well" in c or "Lost Oil" in c]
            show = bdf.drop(columns=["Mulai", "Selesai"]).copy()
            for c in _num:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption("📏 **Compliance = on-time**: sumur ber-deadline di dalam periode yang dites/"
                       "dijadwalkan **pada atau sebelum deadline**. Terjadwal tapi lewat deadline (**Late**) "
                       "dihitung tidak patuh, sama seperti yang tak terjadwal (**Miss**).")

            _p = pd.to_numeric(bdf["Pooled · Compliance, %"], errors="coerce")
            _m = pd.to_numeric(bdf["Mapping Unit · Compliance, %"], errors="coerce")
            _pk = pd.to_numeric(bdf["Pooled · km/well"], errors="coerce")
            _mk = pd.to_numeric(bdf["Mapping Unit · km/well"], errors="coerce")
            _nk = pd.to_numeric(bdf["Manual · km/well"], errors="coerce")
            _nc = pd.to_numeric(bdf["Manual · Compliance, %"], errors="coerce")
            _has_man = _nk.notna().any() or _nc.notna().any()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Rata-rata compliance · Pooled", f"{_p.mean():.1f}%",
                      (f"{(_p - _nc).mean():+.1f} vs Manual" if _nc.notna().any() else None))
            k2.metric("Rata-rata compliance · Mapping", f"{_m.mean():.1f}%", f"{(_m - _p).mean():+.1f} vs Pooled")
            k3.metric("Rata-rata km/well · Pooled", f"{_pk.mean():.2f}",
                      (f"{(_pk - _nk).mean():+.2f} vs Manual" if _nk.notna().any() else None),
                      delta_color="inverse")
            k4.metric("Rata-rata km/well · Mapping", f"{_mk.mean():.2f}", f"{(_mk - _pk).mean():+.2f} vs Pooled",
                      delta_color="inverse")
            if not _has_man:
                st.caption("ℹ️ Kolom Manual kosong — unggah file history di sidebar menu "
                           "**⚖️ Data Komparasi Manual** agar compliance & km/well manual ikut terhitung.")
            else:
                st.caption("ℹ️ Compliance **berbasis on-time**, definisi sama untuk Manual & WELLGO: "
                           "populasi = sumur yang deadline-nya jatuh di dalam periode; **comply** = dites/"
                           "dijadwalkan pada tanggal **≤ deadline**. Sumur yang dijadwalkan tapi **lewat "
                           "deadline (Late)** maupun yang **tak terjadwal (Miss)** sama-sama dihitung "
                           "Not-Comply. Deadline dari kolom min–max sheet kandidat; untuk Manual, tanggal "
                           "tes diambil dari file history. Baris history di luar pool kandidat tak dihitung. "
                           "**PRQ/ORQ** (request prioritas) dikecualikan seperti WELLGO: sekali dites/"
                           "dijadwalkan, kapan pun tanggalnya, dihitung **on-time** (tak pernah Late); yang "
                           "tak dites sama sekali tetap Miss. Tanda PRQ/ORQ WELLGO dari kolom kandidat, "
                           "Manual dari kolom **REASON** file history.")
            if not st.session_state.get("_bench_oil"):
                st.caption("ℹ️ Lost Oil (delta) kosong karena sheet kandidat tak punya kolom laju minyak. "
                           "Tambahkan kolom bernama mis. **NET_OIL** atau **BOPD** lalu jalankan lagi — "
                           "nilainya dihitung sebagai selisih total laju minyak sumur miss-deadline "
                           "antara mode Mapping Unit dan Pooled.")

            # Ekspor memakai header dua baris persis seperti format laporan (Mode / Periode).
            # Ditulis manual, BUKAN lewat kolom MultiIndex: pandas menolak to_excel dgn
            # MultiIndex columns saat index=False (NotImplementedError).
            _flat = bdf.drop(columns=["Mulai", "Selesai", "Kandidat"]).copy()
            _top = ["Mode", "Manual", "", "Pooled (bebas zona)", "", "Mapping Unit", "", ""]
            _sub = ["Periode", "Compliance, %", "km/well", "Compliance, %", "km/well",
                    "Compliance, %", "km/well", "Lost Oil (delta)"]
            # Daftar Not-Comply (Late + Miss), DIPISAH per mode: Manual / Pooled / Mapping Unit.
            _miss = st.session_state.get("_bench_miss")

            def _fmt_nc(df_mode):
                """Rapikan satu tabel Not-Comply: rename kolom & format tanggal ke yyyy-mm-dd."""
                out = df_mode.rename(columns={
                    "well": "Well", "field": "Field", "area": "Area", "subarea": "Sub-area",
                    "category": "Kategori", "unit": "Unit Terakhir", "urgency": "Urgensi (H-)",
                    "status": "Status", "sched_date": "Sched/Test Date",
                    "min_date": "Schedule Date (Min)", "max_date": "Schedule Date (Deadline/Max)"}).copy()
                for _dc in ("Schedule Date (Min)", "Schedule Date (Deadline/Max)", "Sched/Test Date"):
                    if _dc in out.columns:
                        out[_dc] = pd.to_datetime(out[_dc], errors="coerce").dt.strftime("%Y-%m-%d")
                _order = ["Periode", "Status", "Well", "Field", "Area", "Sub-area", "Kategori",
                          "Unit Terakhir", "Schedule Date (Min)", "Schedule Date (Deadline/Max)",
                          "Sched/Test Date", "Urgensi (H-)"]
                out = out[[c for c in _order if c in out.columns]]
                return out.sort_values(["Periode", "Status", "Well"], kind="stable")

            # Tiga mode → tiga tabel & tiga sheet terpisah.
            _mode_sheet = [("Manual", "NotComply-Manual"),
                           ("Pooled (bebas zona)", "NotComply-Pooled"),
                           ("Mapping Unit", "NotComply-Mapping")]
            _nc_by_mode = {}
            if _miss is not None and len(_miss):
                for _mname, _ in _mode_sheet:
                    _mrows = _miss[_miss["Mode"] == _mname]
                    _nc_by_mode[_mname] = _fmt_nc(_mrows.drop(columns=["Mode"])) if len(_mrows) else pd.DataFrame()

            _tot_nc = sum(len(v) for v in _nc_by_mode.values())
            with st.expander(f"📋 Daftar sumur Not-Comply (Late + Miss) — {_tot_nc} baris, dipisah per mode"):
                st.caption("**Late** = dijadwalkan/dites tapi lewat deadline; **Miss** = tak terjadwal sampai "
                           "akhir periode. **Sched/Test Date** = tanggal jadwal/tes (terisi utk Late; kosong "
                           "utk Miss). **Deadline/Max** = tanggal batas. Manual hanya terisi bila file history "
                           "diunggah.")
                for _mname, _ in _mode_sheet:
                    _d = _nc_by_mode.get(_mname, pd.DataFrame())
                    st.markdown(f"**{_mname}** — {len(_d)} sumur Not-Comply")
                    if len(_d):
                        st.dataframe(_d, use_container_width=True, hide_index=True)
                    else:
                        st.caption("  (tidak ada / mode tak terpakai)")

            _bbuf = BytesIO()
            with pd.ExcelWriter(_bbuf, engine="openpyxl") as _bw:
                _flat.to_excel(_bw, sheet_name="Analisis_Performa", index=False, header=False, startrow=2)
                _ws = _bw.sheets["Analisis_Performa"]
                for _j, (_a, _b) in enumerate(zip(_top, _sub), start=1):
                    _ws.cell(row=1, column=_j, value=_a)
                    _ws.cell(row=2, column=_j, value=_b)
                for _c0, _c1 in ((2, 3), (4, 5), (6, 7)):
                    _ws.merge_cells(start_row=1, start_column=_c0, end_row=1, end_column=_c1)
                _ws.freeze_panes = "A3"
                _ws.column_dimensions["A"].width = 14
                for _j in range(2, 9):
                    _ws.column_dimensions[chr(64 + _j)].width = 17
                # Satu sheet Not-Comply per mode (Manual / Pooled / Mapping Unit).
                for _mname, _sheet in _mode_sheet:
                    _d = _nc_by_mode.get(_mname, pd.DataFrame())
                    if len(_d):
                        xl_sheet(_bw, _d, _sheet, _sheet.replace("-", "_"))
            st.download_button("⬇️ Unduh Analisis Performa (.xlsx)", _bbuf.getvalue(),
                               file_name="analisis_performa_manual_vs_wellgo.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_candidates:
    ui.section("Monitor Kandidat — Smart Schedule", eyebrow=f"Semua sumur dari Excel · periode {per_lo_ts.date()} s/d {per_hi_ts.date()}")
    st.caption("Lacak sumur mana dari file kandidat yang **diproses** periode ini vs **ditunda/di luar window**. "
               "Aturan: hanya sumur yang window min–max-nya **overlap** periode yang diprioritaskan; "
               "NW/AWS yang deadline-nya **melewati** akhir periode ditunda (isi celah bila ada sisa kapasitas). "
               "PRQ/ORQ dikecualikan — boleh dijadwalkan sepanjang periode.")

    _sched_map = dict(zip(scheduled_all["well"], zip(scheduled_all["plan_unit"], scheduled_all["day_idx"]))) if len(scheduled_all) else {}
    _miss_w = set(missed["well"]) if len(missed) else set()
    _mout_w = set(missed_outside["well"]) if len(missed_outside) else set()
    _carry_w = set(missed_carry["well"]) if len(missed_carry) else set()
    _elig_w = set(elig_all["well"]) if len(elig_all) else set()
    _off_w  = set(off_wells["well"]) if len(off_wells) else set()

    def _cand_status(r):
        w = r["well"]
        if w in executed: return "✔️ COMP (sudah)"
        if w in pending_set: return "⏳ Pending"
        if w in _sched_map:
            u, di = _sched_map[w]
            return f"✅ Dijadwalkan · {u} · H{int(di)}"
        if (w in _off_w) or (str(r.get("status")).upper() == "OFF"): return "⛔ OFF"
        if w in _carry_w: return f"🚧 {carry_label(ncmp_carry.get(w, ''))}"
        if w in _miss_w: return "⚠️ Miss Deadline (kuota penuh)"
        if w in _mout_w: return "🗓️ Luar periode (deadline lampau)"
        if w in _elig_w:
            if pd.notna(r.get("max_date")) and r["max_date"] > batch_hi:
                return "⏭️ Ditunda (deadline > periode)"
            return "🕓 Antre (tak kebagian)"
        return "➖ Di luar window periode"

    def _kat_lbl(r):
        if r.get("tipe") == "NW": return "NW"
        if r.get("tipe") == "AWS": return "AWS"
        rt = str(r.get("req_tag", "")).upper()
        return rt if rt in ("PRQ", "ORQ") else "RTN"

    cd = raw.copy()
    cd["Status Periode"] = cd.apply(_cand_status, axis=1)
    cd["Kategori"] = cd.apply(_kat_lbl, axis=1)
    cd["Overlap Window"] = np.where((cd["min_date"] <= batch_hi) & (cd["max_date"] >= batch_lo), "✓", "—")

    if unit_map:
        _cov = cd["allow_units"].notna()
        _unmapped = sorted(cd.loc[~_cov, "field"].dropna().astype(str).unique())
        _active_units = set(REMOTE_UNITS[:n_remote]) | set(NONREMOTE_UNITS[:n_nonremote])
        _idle = sorted(({u for us in unit_map.values() for u in us} & set(ALL_UNITS)) - _active_units)
        with st.expander(f"🗺️ Mapping Unit — {int(_cov.sum())}/{len(cd)} sumur terikat daftar unit", expanded=False):
            st.caption(f"Dibaca dari sheet **{SHEET_MAPUNIT}** di file kandidat. Unit di luar daftar tidak "
                       "boleh mengambil sumur lapangan tsb, baik sebagai pembuka klaster maupun saat rute "
                       "ditumbuhkan. Lapangan yang tak ada di sheet dibiarkan bebas seperti mode Pooled.")
            st.dataframe(pd.DataFrame(
                [{"Sub-area": (k[0] or "(semua)"), "Field": k[1], "Unit": ", ".join(v)} for k, v in unit_map.items()]
                ).sort_values(["Sub-area", "Field"]), use_container_width=True, hide_index=True)
            if unitmap_unknown:
                st.warning(f"Unit tak dikenal di sheet (di luar 9 unit MWT): **{', '.join(unitmap_unknown)}** — diabaikan.")
            if _idle:
                st.warning(f"Unit dipetakan tapi TIDAK aktif di slider jumlah unit: **{', '.join(_idle)}** — "
                           "sumur yang hanya boleh digarap unit ini tak akan pernah terjadwal. "
                           "Naikkan slider Unit Area Remote/Non-Remote bila memang perlu.")
            if _unmapped:
                st.caption(f"Lapangan tanpa aturan ({len(_unmapped)}): {', '.join(_unmapped[:25])}"
                           + (" …" if len(_unmapped) > 25 else ""))

    _diproses = cd["Status Periode"].str.startswith("✅")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total kandidat", len(cd))
    c2.metric("Diproses (terjadwal)", int(_diproses.sum()))
    c3.metric("Ditunda / luar window", int(cd["Status Periode"].str.startswith(("⏭️", "➖", "🗓️")).sum()))
    c4.metric("Pure-Miss Deadline", int(cd["Status Periode"].str.startswith("⚠️").sum()),
              help="Deadline di dalam periode tapi kuota kru habis.")
    c5.metric("NCMP-Miss Deadline", int(cd["Status Periode"].str.startswith("🚧").sum()),
              help="Terhalang FACI/ROAD/WOFF dari kolom COMMENT IF NOT COMPLETE, bukan soal kuota kru.")

    f1, f2 = st.columns([2, 2])
    _stat_opts = ["Semua"] + sorted(cd["Status Periode"].unique().tolist())
    pick_stat = f1.selectbox("Filter Status", _stat_opts, key="cand_stat")
    _fld_opts = ["Semua"] + sorted(cd["field"].dropna().astype(str).unique().tolist())
    pick_fld = f2.selectbox("Filter Field", _fld_opts, key="cand_fld")

    view = cd.copy()
    if pick_stat != "Semua": view = view[view["Status Periode"] == pick_stat]
    if pick_fld != "Semua": view = view[view["field"].astype(str) == pick_fld]

    show = view[["well", "field", "area", "subarea", "Kategori", "min_date", "max_date", "Overlap Window", "Status Periode"]].rename(
        columns={"well": "Well", "field": "Field", "area": "Area", "subarea": "Sub-area",
                 "min_date": "Min Date", "max_date": "Max Date (deadline)"}).copy()
    if unit_map:
        # Kolom audit: unit mana saja yang berhak atas sumur ini menurut sheet Mapping_Unit.
        show.insert(4, "Unit Mapping", view["allow_units"].map(lambda u: ", ".join(u) if u else "— (bebas)"))
    show["Min Date"] = pd.to_datetime(show["Min Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
    show["Max Date (deadline)"] = pd.to_datetime(show["Max Date (deadline)"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
    st.dataframe(show.sort_values(["Status Periode", "Field", "Well"]), use_container_width=True, hide_index=True)

    _buf = BytesIO()
    with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
        xl_sheet(_w, show, "Kandidat")
    st.download_button("⬇️ Unduh Monitor Kandidat (.xlsx)", _buf.getvalue(),
                       file_name=f"kandidat_monitor_{per_lo_ts.date()}_{per_hi_ts.date()}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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

        for day_idx, day_date in enumerate(days, 1 + day_offset):
            day_data = scheduled_all[scheduled_all["day_idx"] == day_idx]
            if len(day_data) == 0: continue
            
            ui.day_header(f"Hari Ke-{day_idx}", day_date.strftime("%A, %d %b"), 
                          units=day_data["plan_unit"].nunique(), wells=len(day_data))

            # ── FITUR X-RAY ALGORITMA ──
            day_str = str(day_date.date())
            with st.expander(f"🧠 X-Ray Analisis: Bagaimana Jadwal Hari {day_idx} Terbentuk?", expanded=False):
                if 'audit_logs' in st.session_state and day_str in st.session_state['audit_logs']:
                    st.write("**1. Pemilihan Lapangan & Titik Awal Rute (Anchor)**")
                    # Hapus log duplikat sisaan jika terjadi rerun
                    audit_df = pd.DataFrame(st.session_state['audit_logs'][day_str]).drop_duplicates(subset=["Unit"], keep="first")
                    st.dataframe(audit_df, use_container_width=True, hide_index=True)
                
                # Cek trade-off Miss Deadline vs Sumur jauh
                missed_kritis = missed[(missed["max_date"] <= day_date) & (missed["max_date"] >= batch_lo)] if len(missed) else pd.DataFrame()
                sched_aman = day_data[(day_data["max_date"] - day_date).dt.days >= 3] if len(day_data) else pd.DataFrame()
                
                if len(missed_kritis) > 0 and len(sched_aman) > 0:
                    st.write("---")
                    st.write("🕵️ **Analisis Trade-Off: Mengapa ada sumur Miss Deadline sementara sumur H-4 ikut dites?**")
                    st.warning(f"**Insight Sistem:** Ada **{len(missed_kritis)} sumur krisis (H-0/Overdue)** yang Miss Deadline hari ini, sementara unit MWT mengerjakan **{len(sched_aman)} sumur reguler (H-3 dst)** di lapangan lain.")
                    st.caption(
                        "**Penjelasan:**\n"
                        "1. **Kekalahan Skor Field:** Sumur yang *miss deadline* berada di lapangan yang total skor krisisnya kalah dibanding lapangan pemenang. Armada dikirim ke lapangan yang secara kolektif lebih darurat.\n"
                        "2. **Efisiensi Jarak (Sapu Bersih):** Setelah armada sampai di lapangan pemenang, ia akan 'menyapu' sumur reguler (H-3 dsb) di sekitarnya karena **jaraknya sangat dekat** (< 5 km) dari rute utama (fitur *Elastic Limit*). "
                        "Sistem menolak menjemput sumur krisis yang tertinggal karena lokasinya berada di luar radius efisiensi, yang dapat merusak *Time Budget* harian armada."
                    )
            # ───────────────────────────
            
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
    mode_peta = st.radio("🎛️ Mode Tampilan Peta:", ["📍 Visualisasi Rute & Eksekusi (PyDeck)", "🎯 Seleksi & Assign Massal (Plotly)"], horizontal=True)
    st.divider()

    # =========================================================================
    # 1. FILTER GLOBAL UNTUK KEDUA PETA
    # =========================================================================
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
    lbl2idx = {lbl: i + 1 + day_offset for i, lbl in enumerate(day_labels)}

    c_flt1, c_flt2 = st.columns([3, 1])
    with c_flt1:
        sel_labels = st.multiselect("🗓️ Fokus Tanggal Rute (Pilih 1 untuk view harian aktif)", day_labels, default=day_labels)
        if not sel_labels: sel_labels = day_labels
    with c_flt2:
        dur_pick = st.multiselect("⏱️ Filter Durasi Test", [30, 60], default=[30, 60])
        if not dur_pick: dur_pick = [30, 60]

    sel_idx = sorted(lbl2idx[l] for l in sel_labels)
    is_single_day = len(sel_idx) == 1
    view_day = days[sel_idx[0] - 1 - day_offset] if (is_single_day and len(sel_idx) > 0) else None

    disp = scheduled_all[scheduled_all["day_idx"].isin(sel_idx) & scheduled_all["dur"].isin(dur_pick)].copy() if len(scheduled_all) else scheduled_all.copy()
    
    mco1, mco2, mco3 = st.columns([1.5, 2, 1.4])
    color_mode = mco1.selectbox("🎨 Skema Pewarnaan Peta", ["Otomatis (hari/unit)", "Per kategori (NW/AWS/RTN/PRQ/ORQ)", "Zona remote/non-remote", "Per unit", "Early / Late test"])
    unit_filter = mco2.multiselect("🔧 Batasi Tampilan Unit MWT", sorted(scheduled_all["plan_unit"].dropna().unique().tolist()) if len(scheduled_all) else [])
    search_q = mco3.text_input("🔎 Pencarian Cepat Nama Sumur", placeholder="Contoh: BO083").strip().upper()
    timing_pick = st.multiselect("🕐 Filter Deviasi Window", ["EARLY", "on-time", "LATE", "PRQ", "ORQ"], default=[])
    field_block = st.multiselect("📦 Tampilkan Batas Field Area", field_list, default=[])

    _dl_all = [(per_lo_ts + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(max(1, (per_hi_ts - per_lo_ts).days + 1))]
    _dl_src = raw[raw["has_coord"].fillna(False)].copy() if "has_coord" in raw.columns else raw.iloc[0:0].copy()
    _dl_norm = pd.to_datetime(_dl_src["max_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    _dl_cnt = _dl_norm.value_counts()
    _dl_opt = [f"{d}  ({int(_dl_cnt.get(d, 0))} sumur)" for d in _dl_all]
    _dl_lbl2d = dict(zip(_dl_opt, _dl_all))
    dl_pick_lbl = st.multiselect("🎯 Sorot Sumur ber-Deadline pada Tanggal", _dl_opt, default=[],
                                 help="Tandai di peta semua sumur kandidat yang DEADLINE-nya (max_date) jatuh pada tanggal terpilih.")
    dl_pick = [_dl_lbl2d[l] for l in dl_pick_lbl]
    dl_map = _dl_src[_dl_norm.isin(dl_pick)].copy() if dl_pick else _dl_src.iloc[0:0].copy()
    if dl_pick:
        _sw = set(scheduled_all["well"]) if len(scheduled_all) else set()
        _mw = set(missed["well"]) if len(missed) else set()
        _lw = set(leftover["well"]) if len(leftover) else set()
        _offw = set(off_wells["well"]) if len(off_wells) else set()

        def _dstat(w, mx):
            if w in _sw: return "📅 terjadwal"
            if w in _mw: return "⚠️ miss"
            if w in executed: return "✅ COMP"
            if w in pending_set: return "⏳ PENDING"
            if w in _offw or w in woff_set: return "⛔ OFF"
            if pd.notna(mx) and per_lo_ts <= mx <= per_hi_ts:
                return "⚠️ miss (di luar kandidat)"
            return "🕓 belum" if w in _lw else "➖ tak masuk kandidat"

        dl_map["dl_stat"] = [_dstat(w, mx) for w, mx in zip(dl_map["well"], pd.to_datetime(dl_map["max_date"], errors="coerce"))]
        _brk = " · ".join(f"**{d}**: {int((_dl_norm[dl_map.index] == d).sum())}" for d in dl_pick)
        st.caption(f"🎯 {len(dl_map)} sumur ber-deadline pada tanggal terpilih — {_brk}  ·  " + " · ".join(f"{k}: **{v}**" for k, v in dl_map["dl_stat"].value_counts().items()))

    n_miss_coord = int(missed["has_coord"].fillna(False).sum()) if len(missed) else 0
    show_miss = st.checkbox(f"📌 Tampilkan Miss Deadline di peta ({n_miss_coord} sumur berkoordinat)", value=False)
    miss_map = missed[missed["has_coord"].fillna(False)].copy() if len(missed) else leftover.iloc[0:0]

    fb_wells = field_wells_coord[field_wells_coord["field"].isin(field_block)] if field_block else field_wells_coord.iloc[0:0]

    pmap = disp[disp["has_coord"]].copy() if len(disp) else disp.copy()
    if unit_filter: pmap = pmap[pmap["plan_unit"].isin(unit_filter)]
    if timing_pick: pmap = pmap[pmap["timing"].isin(timing_pick)]
    
    man_pick = []
    prev = leftover[leftover["well"].isin(man_pick) & leftover["has_coord"]].copy() if man_pick and len(leftover) else leftover.iloc[0:0]
    search_terms = [t for t in search_q.replace(",", " ").split() if t]
    search_hits = leftover.iloc[0:0]
    layers = []   

    if search_terms:
        _src = week_df[week_df["has_coord"].fillna(False)].copy() if len(week_df) else pd.DataFrame()
        _sched_pool = set(_src["well"]) if len(_src) else set()
        if "has_coord" in raw.columns:
            _rest = raw[raw["has_coord"].fillna(False) & ~raw["well"].isin(_sched_pool)].copy()
            if len(_rest):
                _dtc_s = [c for c in _src.columns if pd.api.types.is_datetime64_any_dtype(_src[c])] if len(_src) else []
                _src = pd.concat([_src, _rest], ignore_index=True) if len(_src) else _rest
                for _c in _dtc_s:  
                    if _c in _src.columns: _src[_c] = pd.to_datetime(_src[_c], errors="coerce")
        if len(_src):
            _wu = _src["well"].astype(str).str.upper()
            smask = pd.Series(False, index=_src.index)
            for t in search_terms: smask |= _wu.str.contains(t, regex=False)
            search_hits = _src[smask].copy()
        if len(search_hits):
            _outside = int((~search_hits["well"].isin(_sched_pool)).sum())
            if _outside:
                st.caption(f"🔎 {len(search_hits)} sumur ketemu · **{_outside} di luar cakupan penjadwalan saat ini** (tersaring Cakupan Jadwal / COMP / OFF / di luar window) — tetap ditampilkan di peta.")

    TIPE_RING = {"NW": [220, 30, 30], "AWS": [245, 150, 20], "REG": [120, 120, 120]}
    TIMING_COL = {"EARLY": [30, 120, 220], "on-time": [150, 150, 150], "LATE": [220, 30, 30], "PRQ": [150, 80, 200], "ORQ": [0, 160, 140]}
    legend = ""
    
    if len(pmap):
        if color_mode == "Per kategori (NW/AWS/RTN/PRQ/ORQ)":
            KAT_COL = {"NW": [107, 79, 216], "AWS": [230, 178, 58], "RTN": [31, 157, 114], "PRQ": [59, 130, 246], "ORQ": [214, 71, 58]}
            def _kat(r):
                if r.get("tipe") == "NW": return "NW"
                if r.get("tipe") == "AWS": return "AWS"
                rt = str(r.get("req_tag", "")).upper()
                if rt == "PRQ": return "PRQ"
                if rt == "ORQ": return "ORQ"
                return "RTN"
            pmap["katcol"] = pmap.apply(_kat, axis=1)
            pmap["color"] = pmap["katcol"].map(KAT_COL)
            legend = "🟣 NW · 🟠 AWS · 🟢 RTN · 🔵 PRQ · 🔴 ORQ"
        elif color_mode == "Zona remote/non-remote":
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
        def _dcol(name, default="—"):
            if name not in d.columns: return pd.Series(default, index=d.index)
            return pd.to_datetime(d[name], errors="coerce").dt.strftime("%Y-%m-%d").fillna(default)
        d["tgl_str"] = _dcol("plan_day", "belum terjadwal")
        d["min_str"] = _dcol("min_date")
        d["max_str"] = _dcol("max_date")
        if "timing_label" not in d.columns: d["timing_label"] = ""
        d["ket"] = d["timing_label"].fillna("").replace("", "-")
        if "plan_unit" not in d.columns: d["plan_unit"] = "-"
        d["plan_unit"] = d["plan_unit"].fillna("-")
        d["seed_txt"] = np.where(d["is_seed"].fillna(False), " · ★ SEED (anchor rute)", "") if "is_seed" in d.columns else ""
        if "kat_full" in d.columns and d["kat_full"].notna().any():
            d["katfull"] = d["kat_full"].fillna("-")
        else:
            d["katfull"] = [kat_label(c, t, r) for c, t, r in zip(
                d.get("category", pd.Series("", index=d.index)),
                d.get("tipe", pd.Series("", index=d.index)),
                d.get("req_tag", pd.Series("", index=d.index)))]
        return d
        
    pmap = _tipcols(pmap)

    # =========================================================================
    # 2. PERCABANGAN RENDER PETA
    # =========================================================================
    if mode_peta == "📍 Visualisasi Rute & Eksekusi (PyDeck)":
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
        _seg_road = _seg_flat = 0   # segmen rute yang dapat geometri jalan vs jatuh ke garis lurus
        if draw_units:
            if show_block:
                polys = [{"polygon": block_polygon(sub), "color": list(sub["color"].iloc[0]) + [55]} for u, sub in pmap.groupby("plan_unit") if len(sub) >= 3]
                if polys: layers.append(pdk.Layer("PolygonLayer", data=polys, get_polygon="polygon", get_fill_color="color", get_line_color="color", line_width_min_pixels=1, stroked=True, filled=True))
            lines, paths = [], []
            for u, sub in pmap.groupby("plan_unit"):
                s = sub.reset_index(drop=True)
                order, _ = optimize_route(s["lat"].values, s["lon"].values)
                col = list(s["color"].iloc[0])
                if _DRAW_ROAD and len(order) > 1:
                    p = road_route_path(s["lon"].values, s["lat"].values, order,
                                        _osrm_url_cfg, _osrm_to_cfg, _osrm_profile_cfg)
                    if len(p) > 1:
                        paths.append({"path": p, "color": col})
                    # road_route_path mengisi _ROAD_GEOM untuk pasangan yang berhasil diambil.
                    # Segmen yang tetap tak ada di cache = digambar garis lurus (fallback).
                    for a in range(len(order) - 1):
                        i, j = order[a], order[a + 1]
                        if (_rk(s.loc[i, "lat"], s.loc[i, "lon"]), _rk(s.loc[j, "lat"], s.loc[j, "lon"])) in _ROAD_GEOM:
                            _seg_road += 1
                        else:
                            _seg_flat += 1
                for a in range(len(order) - 1):
                    i, j = order[a], order[a + 1]
                    lines.append({"from": [s.loc[i, "lon"], s.loc[i, "lat"]], "to": [s.loc[j, "lon"], s.loc[j, "lat"]], "color": col})
            if _DRAW_ROAD and paths:
                layers.append(pdk.Layer("PathLayer", data=paths, get_path="path", get_color="color", width_min_pixels=3, get_width=4))
            elif lines:
                layers.append(pdk.Layer("LineLayer", data=pd.DataFrame(lines), get_source_position="from", get_target_position="to", get_color="color", get_width=2))

        if len(dl_map):
            dm = _tipcols(dl_map.copy())
            dm["dcol"] = [[214, 39, 140]] * len(dm)
            dm["ket"] = dm["dl_stat"]
            layers.append(pdk.Layer("ScatterplotLayer", data=dm, get_position=["lon", "lat"],
                                    get_fill_color=[214, 39, 140, 40], get_line_color="dcol",
                                    get_radius=340, line_width_min_pixels=3, stroked=True,
                                    filled=True, pickable=True))
            layers.append(pdk.Layer("TextLayer", data=dm, get_position=["lon", "lat"],
                                    get_text="well", get_size=13, get_color=[150, 20, 100],
                                    get_pixel_offset=[0, -22], font_family="Arial",
                                    font_weight="bold", pickable=False))
        
        if len(pmap):
            layers.append(pdk.Layer("ScatterplotLayer", data=pmap, get_position=["lon", "lat"], get_fill_color="color", get_radius="radius", get_line_color="ring", get_line_width="ringw", line_width_min_pixels=1, stroked=True, filled=True, pickable=True, opacity=0.9))
            if "is_seed" in pmap.columns:
                _seeds = pmap[pmap["is_seed"].fillna(False)].copy()
                if len(_seeds):
                    _seeds["mark"] = "★"
                    layers.append(pdk.Layer("TextLayer", data=_seeds, get_position=["lon", "lat"], get_text="mark",
                        get_size=24, get_color=[20, 20, 20], get_pixel_offset=[0, -20],
                        font_family="Arial", font_weight="bold", pickable=False))

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
            mm["tipe"] = mm["cat"]
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
            
        foc = search_hits if len(search_hits) else (miss_map if (show_miss and len(miss_map)) else (dl_map if (len(dl_map) and not len(pmap)) else (pmap if len(pmap) else (fb_wells if (field_block and len(fb_wells)) else leftover.iloc[0:0]))))
        zoom_lvl = 13.5 if len(search_hits) == 1 else (13.0 if (show_miss and len(miss_map)==1) else (13.0 if len(dl_map)==1 else (10.0 if (field_block and len(fb_wells)) else 8.5)))
            
        lat_init = float(foc["lat"].mean()) if len(foc) else 1.6
        lon_init = float(foc["lon"].mean()) if len(foc) else 101.3
        
        view = pdk.ViewState(latitude=lat_init, longitude=lon_init, zoom=zoom_lvl)
        tip = "{well} [{katfull}]{seed_txt} · {ket}\nTanggal Plan: {tgl_str} | Unit: {plan_unit}\nWindow Execution: {min_str} → {max_str}"
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="road", tooltip={"text": tip}))
        miss_note = "  📌 Miss Deadline = pin ring merah + label nama, tipe, & window." if (show_miss and len(miss_map)) else ""
        st.caption(f"💡 {legend}. Ring Merah=NW, Oranye=AWS. **★ = sumur seed** (anchor pertama tiap rute unit). Garis biru menghubungkan sequence rute TSP antar sumur.{miss_note}")
        if _DRAW_ROAD and _seg_flat:
            if _GEOM_FAIL:
                st.warning(f"🛣️ {_seg_flat} dari {_seg_road + _seg_flat} segmen rute digambar **garis lurus** "
                           "karena **server OSRM tak terjangkau** saat render (fetch dihentikan agar peta tak "
                           "menggantung). Cek OSRM Base URL di sidebar, atau host OSRM sendiri untuk hasil "
                           "yang konsisten dan cepat.")
            else:
                st.info(f"🛣️ {_seg_flat} dari {_seg_road + _seg_flat} segmen rute masih **garis lurus** karena "
                        "geometri jalannya belum ter-cache. Geometri diambil bertahap tiap kali peta dirender "
                        f"(maksimum {_GEOM_FETCH_MAX} pasangan per render) lalu disimpan lokal, jadi render "
                        "peta ini beberapa kali sampai semua rute mengikuti jalan. Sekali ter-cache, ganti "
                        "periode tak perlu ambil ulang.")

    else:
        # --- RENDER PLOTLY (Lasso Select) ---
        st.info("💡 **TIPS ZOOM & PAN:** Peta ini di-set supaya bisa di-zoom/pan dengan mouse. Jika ingin memilih sumur, **klik icon Lasso (Tali)** atau **Box Select** di menu pojok kanan atas peta, lalu tarik kursor melingkari sumur.")
        
        import plotly.express as px
        
        def rgb_to_hex(rgb):
            if not isinstance(rgb, (list, tuple, np.ndarray)) or len(rgb) < 3:
                return "#808080"
            return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

        # Siapkan data leftover untuk di lasso
        leftover_map = leftover[leftover['has_coord'] == True].copy() if len(leftover) > 0 else pd.DataFrame()
        
        if not leftover_map.empty:
            # Terapkan Search Filter ke Plotly
            if search_terms:
                leftover_map["hit"] = leftover_map["well"].str.upper().isin(search_terms)
                leftover_map = leftover_map[leftover_map["hit"]]
            
            leftover_map = _tipcols(leftover_map)
            leftover_map["deadline_str"] = pd.to_datetime(leftover_map["max_date"], errors='coerce').dt.strftime('%Y-%m-%d').fillna('-')
            
            # Setup warna fallback (kalau belum punya plan_unit)
            KAT_COL_HEX = {"NW": "#6b4fd8", "AWS": "#e6b23a", "RTN": "#1f9d72", "PRQ": "#3b82f6", "ORQ": "#d6473a"}
            def _kat_hex(r):
                tp = r.get("tipe")
                rt = str(r.get("req_tag", "")).upper()
                if tp == "NW": return "NW"
                if tp == "AWS": return "AWS"
                if rt == "PRQ": return "PRQ"
                if rt == "ORQ": return "ORQ"
                return "RTN"
            
            leftover_map["katcol"] = leftover_map.apply(_kat_hex, axis=1)
            leftover_map["color_px"] = leftover_map["katcol"].map(KAT_COL_HEX).fillna("#808080")

            fig = px.scatter_mapbox(
                leftover_map, 
                lat="lat", lon="lon", 
                hover_name="well",
                hover_data={"lat": False, "lon": False, "field": True, "deadline_str": True, "katfull": True},
                color="color_px",
                color_discrete_map="identity",
                zoom=8.5, height=600
            )
            fig.update_layout(
                mapbox_style="carto-positron", 
                margin={"r":0,"t":0,"l":0,"b":0},
                dragmode="zoom"
            )
            fig.update_traces(marker=dict(size=12, opacity=0.8))

            selection = st.plotly_chart(fig, on_select="rerun", selection_mode=("lasso", "box"), use_container_width=True, key="lasso_map_main")

            if selection and selection["selection"]["points"]:
                selected_indices = [point["point_index"] for point in selection["selection"]["points"]]
                selected_wells_df = leftover_map.iloc[selected_indices]
                selected_well_names = selected_wells_df["well"].tolist()
                
                st.success(f"✅ **{len(selected_well_names)} sumur terpilih!**")
                
                with st.form("lasso_assign_form"):
                    st.write("**Daftar Sumur:**", ", ".join(selected_well_names))
                    la1, la2 = st.columns(2)
                    target_u_lasso = la1.selectbox("Assign ke Unit MWT:", ALL_UNITS, key="lasso_unit")
                    target_d_lasso = la2.selectbox("Pada Hari ke-:", day_nums, key="lasso_day")
                    
                    if st.form_submit_button("🚀 Force Assign (Massal)", type="primary", use_container_width=True):
                        st.session_state.setdefault("manual_assign", {})
                        st.session_state.setdefault("manual_unassign", [])
                        
                        for w in selected_well_names:
                            st.session_state["manual_assign"][w] = {"unit": target_u_lasso, "day_idx": target_d_lasso}
                            if w in st.session_state["manual_unassign"]:
                                st.session_state["manual_unassign"].remove(w)
                        st.rerun()
            else:
                st.info("Gunakan alat Lasso (Tali) atau Kotak di pojok kanan atas peta untuk menyeleksi sisa sumur.")
        else:
            st.info("Tidak ada sisa sumur berkoordinat yang sesuai dengan filter pencarian.")

with tab_cart:
    # ── 🚑 Rescue Miss-Deadline (Tahap 2) ─────────────────────────────────
    ui.section("🚑 Rescue Miss-Deadline (Tahap 2)", eyebrow="Gabungkan sumur miss ke rute existing TERDEKAT (distance-first, overflow ≤8)")
    SOFT_CAP = 8
    _miss_c = int(missed["has_coord"].fillna(True).sum()) if len(missed) else 0
    st.caption(f"**{len(missed)}** sumur miss deadline ({_miss_c} berkoordinat). Tiap sumur **digabung ke rute unit+hari "
               f"yang sudah ada & paling dekat** dalam window-nya (utamakan jarak), kapasitas dilonggarkan sampai "
               f"**{SOFT_CAP}**/unit/hari. Sumur yang jaraknya melebihi batas detour **dibiarkan miss** biar total jarak tidak meledak.")
    _detour_cap = st.slider("Batas jarak ke rute terdekat (km) — sumur lebih jauh dibiarkan miss", 2, 100, 7, key="rescue_detour")

    # ── Toleransi jadwal Late/Early: izinkan tempel ke rute di LUAR window ─────
    _ct1, _ct2 = st.columns(2)
    _tol_early = _ct1.slider("Toleransi Early (hari sebelum Min)", 0, 14, 0, key="rescue_tol_early")
    _tol_late = _ct2.slider("Toleransi Late (hari setelah Max)", 0, 14, 0, key="rescue_tol_late")
    _te, _tl = pd.Timedelta(days=_tol_early), pd.Timedelta(days=_tol_late)
    if _tol_early or _tol_late:
        st.caption("⏱️ Toleransi aktif — sumur miss (termasuk NW/AWS) boleh ditempel ke rute di **luar "
                   "window**-nya sejauh batas ini. **REG A dikecualikan** dan tetap wajib di dalam window. "
                   "Yang lain tetap dites & rute efisien, tapi penempatan Early/Late tercatat **tidak on-time**.")

    def _cand_days_tol(_w):
        """Hari kandidat (day_idx relatif periode) dalam [Min − tolEarly, Max + tolLate].
        REG A DIKECUALIKAN dari toleransi: wajib di dalam window [Min, Max], tak boleh Early/Late."""
        _e, _l = (pd.Timedelta(0), pd.Timedelta(0)) if bool(_w.get("is_reg_a", False)) else (_te, _tl)
        return [di + day_offset for di in range(1, horizon + 1)
                if (pd.isna(_w["min_date"]) or days[di - 1] >= _w["min_date"] - _e)
                and (pd.isna(_w["max_date"]) or days[di - 1] <= _w["max_date"] + _l)]

    def _posisi_jadwal(_w, _di_global):
        """Klasifikasi hari terpilih relatif window ASLI: Dalam window / Early / Late."""
        _d = days[_di_global - day_offset - 1]
        if pd.notna(_w["max_date"]) and _d > _w["max_date"]:
            return f"Late +{(_d - _w['max_date']).days}h"
        if pd.notna(_w["min_date"]) and _d < _w["min_date"]:
            return f"Early −{(_w['min_date'] - _d).days}h"
        return "Dalam window"

    # ── List sumur Miss-Deadline + unit/hari terdekat (read-only) ─────────────
    # Untuk tiap sumur miss, cari rute unit+hari yang SUDAH ada & paling dekat di
    # dalam window-nya (± toleransi, zona dihormati). Info yang dipakai tombol Rescue.
    if len(missed):
        _rt_pts = {}
        for _, _r in scheduled_all.iterrows():
            if bool(_r.get("has_coord", True)) and pd.notna(_r.get("lat")):
                _rt_pts.setdefault((_r["plan_unit"], int(_r["day_idx"])), []).append(
                    (float(_r["lat"]), float(_r["lon"])))
        _near_rows = []
        for _, _w in missed.sort_values(["urgency", "max_date"]).iterrows():
            _pool = REMOTE_UNITS if str(_w.get("area", "")).upper() in REMOTE_AREAS else NONREMOTE_UNITS
            _cand_days = _cand_days_tol(_w)
            _has_c = bool(_w.get("has_coord", True)) and pd.notna(_w.get("lat"))
            _best = None                       # (unit, day, dist)
            for _di in _cand_days:
                for _u in _pool:
                    _pts = _rt_pts.get((_u, _di))
                    if not _pts:
                        continue
                    _d = (float(np.min(haversine_km(_w["lat"], _w["lon"],
                          np.array([p[0] for p in _pts]), np.array([p[1] for p in _pts]))))
                          if _has_c else 0.0)
                    if _best is None or _d < _best[2]:
                        _best = (_u, _di, _d)
            if _best is None:
                _u, _di, _dist, _stat, _pos = "-", "-", "-", "Tak ada rute (± toleransi)", "-"
            elif not _has_c:
                _u, _di, _dist, _stat, _pos = _best[0], _best[1], "-", "Perlu koordinat", _posisi_jadwal(_w, _best[1])
            else:
                _u, _di, _dist = _best[0], _best[1], round(_best[2], 1)
                _stat = "✅ Bisa disisipkan" if _best[2] <= _detour_cap else "⚠️ Terlalu jauh"
                _pos = _posisi_jadwal(_w, _best[1])
            _near_rows.append({
                "Well": _w["well"], "Kategori": _fv(_w.get("category")),
                "Deadline": pd.to_datetime(_w.get("max_date")).strftime("%Y-%m-%d") if pd.notna(_w.get("max_date")) else "-",
                "Urgensi (H-)": int(_w["urgency"]) if pd.notna(_w.get("urgency")) else "-",
                "Unit Terdekat": _u, "Hari": _di, "Posisi": _pos, "Jarak (km)": _dist, "Status": _stat})
        _near_df = pd.DataFrame(_near_rows)
        with st.expander(f"📋 List Miss-Deadline + unit/hari terdekat ({len(_near_df)} sumur)", expanded=True):
            _bisa = int((_near_df["Status"] == "✅ Bisa disisipkan").sum()) if len(_near_df) else 0
            st.caption(f"Rute unit+hari **existing terdekat** dalam window tiap sumur miss (zona dihormati). "
                       f"Status ikut slider batas detour di atas — **{_bisa}** dari {len(_near_df)} bisa disisipkan "
                       "pada batas sekarang. Tombol **Jalankan Rescue** di bawah yang benar-benar menyisipkan.")
            st.dataframe(_near_df, use_container_width=True, hide_index=True)
            _buf_near = BytesIO()
            with pd.ExcelWriter(_buf_near, engine="openpyxl") as _wn:
                xl_sheet(_wn, _near_df, "Miss-Deadline-Terdekat", "MissTerdekat")
            st.download_button("⬇️ Unduh list (.xlsx)", _buf_near.getvalue(),
                               file_name="miss_deadline_unit_terdekat.xlsx", key="dl_miss_near",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
        _rescued, _added_km, _skip_far, _n_late, _n_early = [], 0.0, 0, 0, 0
        for _, _w in missed.sort_values(["urgency", "max_date"]).iterrows():
            _wn = _w["well"]
            _pool = REMOTE_UNITS if str(_w.get("area", "")).upper() in REMOTE_AREAS else NONREMOTE_UNITS
            _cand_days = _cand_days_tol(_w)                # window ± toleransi Late/Early
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
            _pos = _posisi_jadwal(_w, _bd)          # Early/Late relatif window asli
            if _pos.startswith("Late"):
                _n_late += 1
            elif _pos.startswith("Early"):
                _n_early += 1
            _rescued.append(_wn)
        st.session_state["manual_assign"] = _assign
        st.session_state["rescued_wells"] = _rescued
        st.session_state["rescue_added_km"] = round(_added_km, 1)
        st.session_state["rescue_skip_far"] = _skip_far
        st.session_state["rescue_late_early"] = (_n_late, _n_early)
        st.rerun()

    _resc = st.session_state.get("rescued_wells", [])
    if _resc:
        _placed = [w for w in _resc if w in set(scheduled_all["well"])]
        _akm = st.session_state.get("rescue_added_km", 0.0)
        _sf = st.session_state.get("rescue_skip_far", 0)
        _nl, _ne = st.session_state.get("rescue_late_early", (0, 0))
        st.success(f"✅ {len(_placed)} sumur miss tersisipkan · estimasi tambahan jarak **~{_akm} km**"
                   + (f" · {_sf} sumur dilewati (terlalu jauh)" if _sf else "")
                   + (f" · di antaranya **{_nl} Late / {_ne} Early** (di luar window, tercatat tidak on-time)"
                      if (_nl or _ne) else ""))
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

    # Meter per (unit, hari): panjang bar = cnt/max_wells, label x/y selalu tampil.
    # Warna hanya 2 state (normal vs over) — "penuh" sudah kebaca dari panjang bar
    # penuh + ikon ✅, dan teal-vs-hijau gagal uji keterbedaan (ΔE 7, ambang 15).
    _cap = {(u, d): (int(((scheduled_all["plan_unit"] == u) & (scheduled_all["day_idx"] == d)).sum())
                     if len(scheduled_all) else 0)
            for u in ALL_UNITS for d in day_nums}
    _cap_max = max(1, int(max_wells))

    st.markdown("""<style>
    .wg-cap{overflow-x:auto;padding:2px 0 6px}
    .wg-cap-grid{display:grid;gap:14px 8px;min-width:max-content;align-items:center}
    .wg-cap-hd{font-size:11px;font-weight:600;color:#8DA3A9;text-transform:uppercase;
                letter-spacing:.04em;text-align:center;padding-bottom:2px}
    .wg-cap-u{font:600 12px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;color:#0B2027;
              white-space:nowrap;padding-right:4px}
    .wg-cap-c{min-width:62px}
    .wg-cap-track{height:8px;border-radius:4px;background:#DCE4E6;overflow:hidden}
    .wg-cap-fill{height:100%;border-radius:4px}
    .wg-cap-lbl{font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
                text-align:center;margin-top:3px;white-space:nowrap}
    .wg-cap-lg{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:11px;color:#5E7076}
    .wg-cap-sw{display:inline-block;width:20px;height:8px;border-radius:4px;
               vertical-align:middle;margin-right:5px}
    </style>""", unsafe_allow_html=True)

    _h = [f'<div class="wg-cap"><div class="wg-cap-grid" style="grid-template-columns:'
          f'96px repeat({len(day_nums)},minmax(62px,1fr))">', '<div></div>']
    _h += [f'<div class="wg-cap-hd">Hari {d}</div>' for d in day_nums]
    for u in ALL_UNITS:
        _h.append(f'<div class="wg-cap-u">{u}</div>')
        for d in day_nums:
            cnt = _cap[(u, d)]
            pct = min(100, round(100 * cnt / _cap_max))
            # Glyph teks (bukan emoji) supaya warnanya ikut state & sewarna bar —
            # emoji punya warna terkunci, 🟢/✅ hijau bentrok dgn bar teal.
            if cnt == 0:
                col, ink, icon, tip = "transparent", ui.MUTED, "○", "kosong"
            elif cnt > _cap_max:
                col, ink, icon, tip = ui.ORANGE, ui.ORANGE, "▲", f"over {cnt - _cap_max}"
                pct = 100
            elif cnt == _cap_max:
                col, ink, icon, tip = ui.TEAL, ui.TEAL_DEEP, "✓", "penuh"
            else:
                col, ink, icon, tip = ui.TEAL, ui.INK, "●", f"sisa {_cap_max - cnt} slot"
            _h.append(
                f'<div class="wg-cap-c" title="{u} · Hari {d} · {cnt}/{_cap_max} sumur ({tip})">'
                f'<div class="wg-cap-track"><div class="wg-cap-fill" '
                f'style="width:{pct}%;background:{col}"></div></div>'
                f'<div class="wg-cap-lbl" style="color:{ink}">{icon} {cnt}/{_cap_max}</div></div>')
    _h.append("</div></div>")
    _h.append(
        f'<div class="wg-cap-lg">'
        f'<span><i class="wg-cap-sw" style="background:{ui.TEAL}"></i>'
        f'<b style="color:{ui.INK}">●</b> terisi · '
        f'<b style="color:{ui.TEAL_DEEP}">✓</b> penuh ({_cap_max}/hari)</span>'
        f'<span><i class="wg-cap-sw" style="background:{ui.ORANGE}"></i>'
        f'<b style="color:{ui.ORANGE}">▲</b> lewat kapasitas</span>'
        f'<span><i class="wg-cap-sw" style="background:#DCE4E6"></i>○ kosong</span></div>')
    st.markdown("".join(_h), unsafe_allow_html=True)

    grid_df = pd.DataFrame(
        {f"Hari {d}": [("kosong" if _cap[(u, d)] == 0 else
                        f"⚠️ {_cap[(u, d)]}/{_cap_max} (Over)" if _cap[(u, d)] > _cap_max else
                        f"✅ {_cap[(u, d)]}/{_cap_max}" if _cap[(u, d)] == _cap_max else
                        f"🟢 {_cap[(u, d)]}/{_cap_max}") for u in ALL_UNITS] for d in day_nums},
        index=ALL_UNITS)
    with st.expander("📋 Lihat sebagai tabel"):
        st.dataframe(grid_df, use_container_width=True)

    st.divider()
    
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
                "Hari ke-": st.column_config.NumberColumn("Hari", min_value=day_nums[0], max_value=day_nums[-1]),
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

    with st.expander("🧭 Anchor Rute Manual — tentukan titik awal rute unit"):
        st.caption("Sumur anchor **ditanam lebih dulu** sebagai titik awal klaster, lalu rute unit "
                   "ditumbuhkan mengelilinginya. Beda dgn Force Assign yang menempelkan sumur "
                   "*setelah* optimasi — anchor **membentuk** rutenya. Aturan zona & alokasi "
                   "mutlak (GP/BENAR/Balam_South) tetap menang; anchor yang melanggar diabaikan.")
        _anch = dict(st.session_state.get("route_anchors", {}))
        _pool_anchor = sorted(set(elig["well"])) if len(elig) else []
        ac1, ac2 = st.columns([3, 2])
        with ac1:
            anch_pick = st.multiselect("Pilih sumur jadi anchor", _pool_anchor, key="anch_pick")
        with ac2:
            b1, b2 = st.columns(2)
            anch_unit = b1.selectbox("Unit", ALL_UNITS, key="anch_unit")
            anch_day = b2.selectbox("Hari ke-", day_nums, key="anch_day")
            if st.button("⚓ Jadikan Anchor", use_container_width=True,
                         type="primary", disabled=not anch_pick):
                _anch.update({w: {"unit": anch_unit, "day_idx": int(anch_day)} for w in anch_pick})
                st.session_state["route_anchors"] = _anch
                st.rerun()
        if _anch:
            st.write(f"**{len(_anch)} anchor aktif:**")
            for w, info in list(_anch.items()):
                q1, q2 = st.columns([5, 1])
                _hit = scheduled_all[scheduled_all["well"] == w] if len(scheduled_all) else scheduled_all
                if len(_hit):
                    _r = _hit.iloc[0]
                    _ok = (_r["plan_unit"] == info["unit"] and int(_r["day_idx"]) == int(info["day_idx"]))
                    _st = ("✅ jadi seed" if _r.get("is_seed") else "⚠️ terjadwal tapi bukan seed") if _ok \
                          else f"⚠️ dipindah ke {_r['plan_unit']} H{int(_r['day_idx'])} (zona/alokasi mutlak)"
                else:
                    _st = "❌ tak terjadwal (di luar window hari itu?)"
                q1.write(f"• **{w}** → {info['unit']} · Hari {info['day_idx']} — {_st}")
                if q2.button("🗑️", key=f"del_anch_{w}"):
                    _anch.pop(w, None); st.session_state["route_anchors"] = _anch; st.rerun()
            if st.button("Hapus semua anchor", key="clr_anch"):
                st.session_state["route_anchors"] = {}; st.rerun()

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
            man_day = a2.selectbox("Hari ke-", day_nums, key="man_day")
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

with tab_matrix:
    ui.section("🗓️ Matriks Deadline per Tanggal", eyebrow="Sumur jatuh tempo (deadline) dikelompokkan per tanggal")
    _dl = cand[cand["max_date"].notna()].copy()
    if len(_dl):
        _dl["is_nwaws"] = _dl["is_nwaws"].fillna(False)
        _tu = _dl["tipe"].astype(str).str.upper()
        _rt = _dl["req_tag"].astype(str).str.upper()
        _dl["Tipe"] = np.select(
            [_dl["is_nwaws"] & (_tu == "NW"), _dl["is_nwaws"] & (_tu == "AWS"),
             _rt == "PRQ", _rt == "ORQ"],
            ["NW", "AWS", "PRQ", "ORQ"], default="Regular")
        _dl["Deadline"] = _dl["max_date"].dt.strftime("%Y-%m-%d")
        _dl["Area"] = _dl["area"].fillna("-")

        c1, c2, c3 = st.columns([2.4, 2.4, 1.8])
        _types = ["NW", "AWS", "PRQ", "ORQ", "Regular"]
        sel_t = c1.multiselect("Filter Tipe", _types, default=_types, key="dlmx_t")
        _areas = sorted(_dl["Area"].unique())
        sel_a = c2.multiselect("Filter Area", _areas, default=_areas, key="dlmx_a")
        by = c3.radio("Kolom matriks", ["Tipe", "Area", "Field"], key="dlmx_by", horizontal=True)
        cc1, cc2 = st.columns([2, 2])
        in_period = cc1.checkbox(f"Hanya deadline dalam periode terpilih ({per_lo_ts.date()} s/d {per_hi_ts.date()})",
                                 value=True, key="dlmx_period")
        show_names = cc2.toggle("Tampilkan nama sumur di dalam sel (bukan jumlah)", value=False, key="dlmx_names")

        v = _dl[_dl["Tipe"].isin(sel_t) & _dl["Area"].isin(sel_a)]
        if in_period:
            v = v[(v["max_date"] >= per_lo_ts) & (v["max_date"] <= per_hi_ts)]
        if len(v):
            col_field = {"Tipe": "Tipe", "Area": "Area", "Field": "field"}[by]
            if show_names:
                piv = (v.groupby(["Deadline", col_field])["well"]
                         .apply(lambda s: ", ".join(sorted(s))).unstack(fill_value=""))
                piv = piv.reindex(sorted(piv.index))
                st.dataframe(piv, use_container_width=True)
            else:
                piv = v.pivot_table(index="Deadline", columns=col_field, values="well",
                                    aggfunc="count", fill_value=0)
                piv = piv.reindex(sorted(piv.index))
                piv["TOTAL"] = piv.sum(axis=1)
                piv.loc["TOTAL"] = piv.sum(axis=0)
                st.dataframe(piv, use_container_width=True)
            st.caption(f"**{len(v)}** sumur punya deadline"
                       + (f" dalam periode {per_lo_ts.date()}–{per_hi_ts.date()}" if in_period else " (semua tanggal)")
                       + f". Baris = tanggal deadline (Latest Date), kolom = {by}. "
                       f"Sumber: pool kandidat aktif (COMP/PENDING/OFF sudah dikecualikan).")

            with st.expander("📋 Rincian sumur per deadline"):
                det = v[["Deadline", "well", "Tipe", "field", "Area", "min_date", "max_date", "unit", "status"]].copy()
                det["min_date"] = det["min_date"].dt.strftime("%Y-%m-%d").fillna("-")
                det["max_date"] = det["max_date"].dt.strftime("%Y-%m-%d").fillna("-")
                det = det.rename(columns={"well": "Well", "field": "Field", "min_date": "Earliest",
                                          "max_date": "Latest (Deadline)", "unit": "Unit Terakhir", "status": "Status"})
                st.dataframe(det.sort_values(["Deadline", "Tipe", "Well"]), use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada sumur yang cocok dengan filter.")
    else:
        st.info("Belum ada sumur berdeadline di pool kandidat.")

    st.divider()
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
        ui.section("Daftar Pure-Miss Deadline (Kapasitas Penuh)", eyebrow="Butuh aksi manual/tambah shift")
        st.dataframe(missed[["well", "unit", "subarea", "category", "urgency", "max_date"]].rename(columns={"max_date": "deadline", "unit": "unit_asli"}).sort_values("urgency"), use_container_width=True, hide_index=True)

        with st.expander("🔎 Review Miss Deadline vs SCH_Database — tandai COMP manual", expanded=False):
            _comp_review_panel(missed, key="rev_miss")

    if len(missed_outside):
        ui.section(f"Miss Deadline — Window di Luar Periode ({len(missed_outside)})",
                   eyebrow="Deadline sudah lewat sebelum periode mulai · bukan gagal kapasitas")
        st.caption("Sumur ini rentang min–max-nya berada di luar periode terpilih (deadline sudah terlewat "
                   "sebelum periode dimulai). Dipisahkan dari Miss Deadline karena bukan kasus kru penuh — "
                   "tinjau manual apakah perlu dijadwalkan susulan atau di-exclude.")
        st.dataframe(
            missed_outside[["well", "unit", "subarea", "category", "min_date", "max_date"]]
            .rename(columns={"min_date": "earliest", "max_date": "deadline", "unit": "unit_asli"})
            .sort_values("deadline"),
            use_container_width=True, hide_index=True)

    if len(missed_carry):
        ui.section(f"{NCMP_CARRY_LABEL} ({len(missed_carry)})",
                   eyebrow="NCMP FACI/ROAD/WOFF · sudah dibawa sepanjang periode, tetap tak kebagian slot")
        st.caption("Sumur ini NCMP karena **hambatan lapangan** (COMMENT IF NOT COMPLETE = FACI/ROAD/WOFF), "
                   "sudah dijadwal ulang di sepanjang sisa periode walau deadline-nya lewat, dan sampai akhir "
                   "periode tetap tak dapat slot. **Tidak dihitung sebagai Miss Deadline** karena penyebabnya "
                   "bukan kuota kru — perlu tindak lanjut fasilitas/akses/status sumur lebih dulu.")
        st.dataframe(
            missed_carry[["well", "kategori_ncmp", "kode_hambatan", "unit", "subarea", "category", "min_date", "max_date"]]
            .rename(columns={"kategori_ncmp": "kategori", "kode_hambatan": "kode", "min_date": "earliest",
                             "max_date": "deadline", "unit": "unit_asli"})
            .sort_values(["kode", "deadline"]),
            use_container_width=True, hide_index=True)

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
    tot_expired = len(ncmp_expired)
    _bal = tot_ncmp - tot_replan - tot_woff - tot_nodata - tot_expired
    st.caption(
        f"**Rekonsiliasi NCMP:** Total {tot_ncmp} = {tot_replan} dijadwal ulang + {tot_woff} OFF/skip + "
        f"**{tot_nodata} tidak ada baris kandidat** (ke-exclude area mis. LIBO, filter MPAS-only, "
        f"atau memang tak ada di sheet Kandidat)"
        + (f" + **{tot_expired} window kedaluwarsa**" if tot_expired else "")
        + (f" + {_bal} lainnya" if _bal else "") + ".")
    if tot_nodata:
        with st.expander(f"🔻 {tot_nodata} NCMP tanpa baris kandidat (tidak bisa di-replan)"):
            st.dataframe(pd.DataFrame({"well": ncmp_no_data}), use_container_width=True, hide_index=True)
    if tot_expired:
        with st.expander(f"⌛ {tot_expired} NCMP dgn window sudah lewat periode (tidak dijadwalkan)"):
            st.caption("Window min–max sumur ini tak lagi overlap periode terpilih. Hanya PRQ/ORQ dan NCMP "
                       "ber-COMMENT FACI/ROAD/WOFF yang boleh dijadwalkan saat overdue — terbitkan ulang sbg "
                       "request bila tetap perlu dites.")
            st.dataframe(expired_df.rename(columns={"min_date": "min", "max_date": "max (deadline)"}),
                         use_container_width=True, hide_index=True)

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
            _rp = replan_df.copy()
            _rp["hasil"] = np.where(_rp["well"].isin(sched_wells), "📅 terjadwal",
                             np.where(_rp["well"].isin(set(missed_carry["well"]) if len(missed_carry) else set()),
                                      "🚧 " + _rp["kode_hambatan"].map(carry_label).fillna(NCMP_CARRY_LABEL),
                                      "🕓 belum kebagian"))
            st.dataframe(_rp.rename(columns={"plan_date": "tgl_NCMP", "reason": "alasan",
                                             "comment": "comment_if_not_complete", "kode_hambatan": "kode"}),
                         use_container_width=True, hide_index=True)
            st.caption("Kode **FACI/ROAD/WOFF** dibaca dari kolom *COMMENT IF NOT COMPLETE*: sumur ini tetap "
                       "dijadwalkan ulang sepanjang sisa periode walau deadline (max_date) sudah lewat. Bila "
                       f"sampai akhir periode tak kebagian slot, statusnya *{NCMP_CARRY_LABEL}*, bukan Miss Deadline.")
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
        # Header kolom dari tanggal yang sudah dibawa tiap baris — JANGAN days[Hari-1]:
        # "Hari" itu day_idx relatif AWAL periode, jadi saat mulai planning bukan hari-1
        # indeksnya lewat dari panjang days.
        _h2d = dict(zip(kdf_an["Hari"], kdf_an["Tanggal"]))
        piv_an.columns = [pd.Timestamp(_h2d[c]).strftime("%Y-%m-%d") for c in piv_an.columns]
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

    # Well-OFF export: SEMUA sumur OFF yang window min-max-nya overlap periode (eligible untuk
    # dites tapi sumurnya mati), termasuk yang tersaring dari kandidat karena ter-exclude
    # pending/executed di SCH. off_wells (dipakai KPI dashboard) hanya memuat OFF yang lolos gate
    # kandidat, sehingga sumur OFF eligible yang ter-exclude (mis. PENDING di SCH_Database) tidak
    # ikut terekspor. Yang sudah executed (benar-benar dites) tetap dikecualikan.
    _off_export = raw[(raw["status"] == "OFF")
                      & (raw["min_date"] <= batch_hi) & (raw["max_date"] >= batch_lo)
                      & (~raw["well"].isin(executed))].copy()
    _off_cols = [c for c in ["well", "unit", "subarea", "field", "category", "kat_full",
                             "dur", "status", "min_date", "max_date"] if c in _off_export.columns]
    _off_ren = {"unit": "unit_asli", "dur": "durasi_test_menit", "kat_full": "kategori",
                "min_date": "earliest", "max_date": "deadline", "status": "status_sumur"}

    ex1, ex2 = st.columns(2)
    out_w = BytesIO()
    with pd.ExcelWriter(out_w, engine="openpyxl") as w:
        xl_sheet(w, scheduled_all[exp_cols].rename(columns=ren).sort_values(["hari", "grup", "urgency"]), "Jadwal_Mingguan")
        if len(missed): xl_sheet(w, missed[["well", "unit", "subarea", "category", "dur", "urgency", "max_date"]].rename(columns={"dur": "durasi_test_menit", "max_date": "deadline", "unit": "unit_asli"}), "Miss-Deadline", "MissDeadline")
        if len(missed_outside): xl_sheet(w, missed_outside[["well", "unit", "subarea", "category", "dur", "min_date", "max_date"]].rename(columns={"dur": "durasi_test_menit", "min_date": "earliest", "max_date": "deadline", "unit": "unit_asli"}), "Luar-Periode", "LuarPeriode")
        if len(missed_carry): xl_sheet(w, missed_carry[["well", "kategori_ncmp", "kode_hambatan", "unit", "subarea", "category", "dur", "min_date", "max_date"]].rename(columns={"kategori_ncmp": "kategori", "kode_hambatan": "kode", "dur": "durasi_test_menit", "min_date": "earliest", "max_date": "deadline", "unit": "unit_asli"}), "NCMP-Hambatan", "NCMPHambatan")
        if len(_off_export): xl_sheet(w, _off_export[_off_cols].rename(columns=_off_ren), "Well-OFF", "WellOFF")
    ex1.download_button("⬇️ Unduh Master Mingguan (.xlsx)", out_w.getvalue(), file_name=f"jadwal_mingguan_{week_lo.date()}_{week_hi.date()}.xlsx", mime=XLSX_MIME)

    if view_day is not None:
        out_d = BytesIO()
        # Sumur OFF yang RELEVAN utk hari ini = window min-max-nya mencakup tanggal itu,
        # jadi seharusnya bisa dites hari ini tapi sumurnya mati. Bukan seluruh daftar OFF.
        _od = _off_export.copy()
        if len(_od):
            _cov = ((_od["min_date"].isna() | (_od["min_date"] <= view_day))
                    & (_od["max_date"].isna() | (_od["max_date"] >= view_day)))
            _od = _od[_cov]
        with pd.ExcelWriter(out_d, engine="openpyxl") as w:
            xl_sheet(w, disp[exp_cols].rename(columns=ren).sort_values(["grup", "urgency"]), "Jadwal_Harian")
            xl_sheet(w, unit_summary(disp, speed), "Ringkasan_Unit")
            if len(_od): xl_sheet(w, _od[_off_cols].rename(columns=_off_ren), "Well-OFF", "WellOFFHarian")
        ex2.download_button(f"⬇️ Unduh Rute Harian {view_day.date()} (.xlsx)", out_d.getvalue(), file_name=f"jadwal_harian_{view_day.date()}.xlsx", mime=XLSX_MIME, type="primary")
    else:
        ex2.caption("Pilih **1 tanggal tunggal** di tab Peta Rute untuk mengaktifkan unduhan rute harian.")

    st.divider()
    ui.section("Export ke Google Maps (KML)", eyebrow="Titik sumur + garis rute per unit — buka di Google My Maps / Earth")
    _kml_src = scheduled_all[scheduled_all["has_coord"].fillna(False)].copy() if len(scheduled_all) else scheduled_all
    if len(_kml_src):
        km1, km2 = st.columns(2)
        _kml_all = build_kml(_kml_src, title=f"WELLGO {week_lo.date()}–{week_hi.date()}")
        km1.download_button("🌍 Unduh Semua Rute (.kml)", _kml_all,
                            file_name=f"wellgo_rute_{week_lo.date()}_{week_hi.date()}.kml",
                            mime="application/vnd.google-earth.kml+xml")
        if view_day is not None:
            _kd = _kml_src[_kml_src["plan_day"] == view_day]
            if len(_kd):
                km2.download_button(f"🌍 Unduh Rute {view_day.date()} (.kml)",
                                    build_kml(_kd, title=f"WELLGO {view_day.date()}"),
                                    file_name=f"wellgo_rute_{view_day.date()}.kml",
                                    mime="application/vnd.google-earth.kml+xml", type="primary")
        else:
            km2.caption("Pilih 1 tanggal di tab Peta untuk unduh rute harian.")
        st.caption("**Cara buka di Google Maps:** buka [Google My Maps](https://mymaps.google.com) → *Create a new map* "
                   "→ *Import* → unggah file `.kml` ini. Titik = sumur (warna per unit), garis = urutan rute. "
                   "Atau buka langsung di **Google Earth**.")
    else:
        st.info("Belum ada rute terjadwal untuk diekspor.")

    st.divider()
    ui.section("Buka Rute Langsung di Google Maps", eyebrow="Tanpa upload KML — klik untuk langsung navigasi per unit")
    if view_day is not None and len(_kml_src):
        _gd = _kml_src[_kml_src["plan_day"] == view_day]
        if len(_gd):
            st.caption(f"Rute hari **{view_day.date()}** per unit. Setiap tautan membuka Google Maps dengan sumur "
                       "terurut sebagai titik perjalanan, langsung siap navigasi tanpa upload apa pun. Urutannya "
                       "sama dengan rute TSP pada peta.")
            for u, sub in _gd.groupby("plan_unit"):
                s = sub.reset_index(drop=True)
                if not len(s):
                    continue
                order, _ = optimize_route(s["lat"].values, s["lon"].values)
                pts = [f'{s.loc[i, "lat"]:.6f},{s.loc[i, "lon"]:.6f}' for i in order]
                if len(pts) == 1:
                    url = f"https://www.google.com/maps/search/?api=1&query={pts[0]}"
                else:
                    # URL path-style /maps/dir/ menampung banyak titik dan langsung merutekan.
                    url = "https://www.google.com/maps/dir/" + "/".join(pts)
                _note = ""
                if len(pts) > 10:
                    _note = " · ⚠️ Google Maps membatasi ±10 titik untuk navigasi; sisanya mungkin terpotong"
                st.markdown(f"- **{u}** ({len(pts)} titik) → [Buka rute di Google Maps]({url}){_note}")
        else:
            st.caption("Tidak ada rute untuk tanggal ini.")
    else:
        st.caption("Pilih **1 tanggal tunggal** di tab Peta Rute untuk membuat tautan rute Google Maps per unit.")

    st.divider()
    ui.section("Peta Rute Berlabel (HTML) — buka di browser, tanpa upload",
               eyebrow="Tiap titik menampilkan nama Well — tanpa My Maps, tanpa akun Google")
    if len(_kml_src):
        h1, h2 = st.columns(2)
        h1.download_button("🗺️ Unduh Peta HTML (semua rute)",
                           build_route_html(_kml_src, title=f"WELLGO {week_lo.date()}–{week_hi.date()}").encode("utf-8"),
                           file_name=f"wellgo_peta_{week_lo.date()}_{week_hi.date()}.html", mime="text/html")
        if view_day is not None:
            _hd = _kml_src[_kml_src["plan_day"] == view_day]
            if len(_hd):
                h2.download_button(f"🗺️ Unduh Peta HTML {view_day.date()}",
                                   build_route_html(_hd, title=f"WELLGO {view_day.date()}").encode("utf-8"),
                                   file_name=f"wellgo_peta_{view_day.date()}.html", mime="text/html", type="primary")
        else:
            h2.caption("Pilih 1 tanggal di tab Peta untuk peta HTML harian.")
        st.caption("Buka file `.html` ini di browser mana pun. Tiap titik menampilkan **nomor urut + nama sumur** "
                   "sebagai label tetap; klik titik untuk lihat kategori, durasi, unit, dan deadline. Garis rute "
                   "per unit mengikuti urutan yang sama dengan aplikasi. Tanpa upload, tanpa akun Google "
                   "(butuh internet untuk memuat peta dasar).")
    else:
        st.info("Belum ada rute terjadwal untuk peta HTML.")

with tab_compare:
    ui.section("Komparasi Rute: Manual (History) vs WELLGO", eyebrow="Evaluasi Efisiensi Jarak & Distribusi Harian")

    # Rute manual berasal dari file berformat SCHDatabase yang diunggah di menu Data Komparasi
    # Manual. File itu HANYA dibaca di sini: tidak masuk execution_log, jadi statusnya tak ikut
    # menentukan COMP/NCMP/PENDING maupun kelayakan penjadwalan.
    # Hanya unit MPAS (MP…) yang dibandingkan — unit TS tak punya padanan di sisi WELLGO.
    sch_hist = sch_history([f.getvalue() for f in hist_files]) if hist_files else pd.DataFrame()

    if not len(sch_hist):
        st.info("💡 Unggah file **History (format SCHDatabase)** di sidebar menu **⚖️ Data Komparasi Manual** "
                "untuk melihat perbandingan head-to-head. Kolom yang dibaca: WELL, UNIT, "
                "SCHEDULE_DATE_TEST. File ini murni sumber riwayat rute manual — status di dalamnya "
                "tidak dipakai untuk keputusan COMP/NCMP/PENDING."
                + (" File yang diunggah tidak memuat baris unit MPAS yang bisa dibaca." if hist_files else ""))
    else:
        try:
            # Tambahan: Filter Tanggal khusus tab komparasi
            day_labels_comp = [days[i].strftime("%Y-%m-%d") for i in range(horizon)]
            lbl2idx_comp = {lbl: i + 1 for i, lbl in enumerate(day_labels_comp)}

            # Komparasi hanya bermakna pada tanggal yang dipunyai KEDUA sisi: file riwayat
            # punya jadwal manualnya, WELLGO punya rencananya. Default ke irisan itu.
            _sch_days = set(sch_hist["date"].dt.strftime("%Y-%m-%d"))
            _both = [l for l in day_labels_comp if l in _sch_days]
            st.caption(f"🗃️ Sumber rute manual: **{len(hist_files)} file history** — {len(sch_hist)} baris "
                       f"unit MPAS pada {len(_sch_days)} tanggal. Dibaca untuk komparasi saja, "
                       "tidak masuk SCH_Database. "
                       + (f"Beririsan dengan horizon WELLGO di **{len(_both)}** tanggal."
                          if _both else "⚠️ **Tidak ada tanggal yang beririsan** dengan horizon WELLGO — "
                          "geser Mulai Perencanaan ke tanggal yang ada riwayatnya agar sebanding."))

            c_flt_comp, _ = st.columns([3, 1])
            with c_flt_comp:
                comp_sel_labels = st.multiselect("🗓️ Fokus Tanggal Rute (Pilih untuk view komparasi)",
                                                 day_labels_comp, default=(_both or day_labels_comp),
                                                 key="comp_sel_dates")

            if not comp_sel_labels:
                comp_sel_labels = _both or day_labels_comp

            comp_sel_idx = sorted(lbl2idx_comp[l] for l in comp_sel_labels)

            # 1. Jadwal manual = riwayat dari file history, dipetakan ke nama kolom lama
            #    (WELL/UNIT/SCHEDULE DATE) supaya kalkulasi & peta di bawah tak perlu berubah.
            selected_dates = pd.to_datetime(comp_sel_labels).date
            man_df = sch_hist[sch_hist["date"].dt.date.isin(selected_dates)].rename(
                columns={"well": "WELL", "unit": "UNIT", "date": "SCHEDULE DATE"}).copy()
            # Status hanya jadi keterangan di tooltip peta, bukan bahan keputusan apa pun.
            man_df["timing_label"] = man_df["stat"].replace("", "(tanpa status)")

            if man_df.empty:
                st.warning(f"File history tidak punya jadwal unit MPAS pada tanggal {', '.join(comp_sel_labels)}.")
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
                _man_nocoord = sorted(set(man_df["WELL"]) - set(man_valid["WELL"]))
                
                # 2. Kalkulasi Jarak Manual
                manual_km = 0.0
                for (date, unit), group in man_valid.groupby(["SCHEDULE DATE", "UNIT"]):
                    manual_km += route_distance(group["LAT"].values, group["LON"].values)
                
                # Data WELLGO khusus untuk komparasi berdasarkan filter
                comp_disp = scheduled_all[scheduled_all["day_idx"].isin(comp_sel_idx)].copy() if len(scheduled_all) else scheduled_all.copy()
                
                # Kalkulasi Jarak WELLGO DINAMIS
                wellgo_km = 0.0
                # Penyebut km/well HARUS setara: sisi manual hanya menghitung sumur berkoordinat,
                # jadi sisi WELLGO pun begitu. Kalau tidak, sumur tanpa koordinat (0 km) ikut
                # membagi dan WELLGO tampak lebih hemat dari kenyataannya.
                wellgo_wells = int(comp_disp["has_coord"].fillna(False).sum()) if len(comp_disp) else 0
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
                
                st.caption(f"⚖️ Dibandingkan hanya sumur **berkoordinat** di kedua sisi "
                           f"({man_wells} manual · {wellgo_wells} WELLGO) agar km/well setara."
                           + (f" {len(_man_nocoord)} sumur history tanpa koordinat dikecualikan: "
                              + ", ".join(_man_nocoord[:10]) + (" …" if len(_man_nocoord) > 10 else "")
                              if _man_nocoord else ""))
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
                        
                        if "katfull" not in df_map.columns:
                            df_map["katfull"] = (df_map["kat_full"] if "kat_full" in df_map.columns
                                                 else df_map["well"].map(_raw_dedup["kat_full"])).fillna("-")
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

                        lines, paths = [], []
                        for u, sub in df_map.groupby(unit_col):
                            s = sub.reset_index(drop=True)
                            if len(s) > 1:
                                order, _ = optimize_route(s[lat_col].values, s[lon_col].values)
                                col = list(s["color"].iloc[0])
                                if _DRAW_ROAD:
                                    p = road_route_path(s[lon_col].values, s[lat_col].values, order,
                                                        _osrm_url_cfg, _osrm_to_cfg, _osrm_profile_cfg)
                                    if len(p) > 1:
                                        paths.append({"path": p, "color": col})
                                for a in range(len(order) - 1):
                                    i, j = order[a], order[a + 1]
                                    lines.append({
                                        "from": [s.loc[i, lon_col], s.loc[i, lat_col]],
                                        "to": [s.loc[j, lon_col], s.loc[j, lat_col]],
                                        "color": col
                                    })
                        if _DRAW_ROAD and paths:
                            layers.append(pdk.Layer(
                                "PathLayer", data=paths, get_path="path", get_color="color",
                                width_min_pixels=3, get_width=4
                            ))
                        elif lines:
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
                    
                    tip = "{well} [{katfull}] · {ket}\nTanggal Plan: {tgl_str} | Unit: {plan_unit}\nWindow Execution: {min_str} → {max_str}"
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
            st.error(f"Gagal memproses file history: {str(e)}. Pastikan file berformat SCHDatabase "
                     "dan punya kolom WELL, UNIT, serta SCHEDULE_DATE_TEST yang terbaca.")

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
        _miss_out_w = set(missed_outside["well"]) if len(missed_outside) else set()
        _carry_w = set(missed_carry["well"]) if len(missed_carry) else set()
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
            if w in _carry_w: return f"🚧 NCMP {ncmp_carry.get(w, '')}-Miss Deadline"
            if w in _miss_w: return "⚠️ Miss Deadline"
            if w in _miss_out_w: return "🗓️ Luar Periode"
            if w in _left_w: return "🕓 Antre"
            return "➖ Luar window/exclude"

        pri["Kategori"] = [_kat(r) for _, r in pri.iterrows()]
        pri["Status"] = [_stat(w) for w in pri["well"]]
        cats = ["NW", "AWS", "PRQ", "ORQ"]
        pick = st.multiselect("Filter kategori", cats, default=cats, key="pri_cat")
        view = pri[pri["Kategori"].isin(pick)].copy()
        _cc = view["Kategori"].value_counts()
        _bi_v = view["is_breakin"].fillna(False) if "is_breakin" in view.columns else pd.Series(False, index=view.index)
        st.caption(" · ".join(f"**{k}**: {int(_cc.get(k, 0))}" for k in cats)
                   + f"  ·  total: **{len(view)}**"
                   + (f"  ·  dari sheet Break-In: **{int(_bi_v.sum())}**" if int(_bi_v.sum()) else ""))
        if len(view):
            _onoff = view["status"].apply(lambda s: "🔴 OFF" if str(s).upper().strip() == "OFF" else "🟢 ON") if "status" in view.columns else "🟢 ON"
            pri_disp = view[["well", "Kategori", "category", "field", "area", "min_date", "max_date", "Status"]].copy()
            pri_disp.insert(2, "ON/OFF", _onoff)
            # Sumur sisipan dari sheet BreakIn ikut di sini (mereka memang sudah masuk `raw`),
            # tapi tanpa penanda tak terbedakan dari kandidat reguler.
            pri_disp.insert(3, "Break-In", np.where(_bi_v.values, "✅", ""))
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