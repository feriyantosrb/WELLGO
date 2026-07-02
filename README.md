# WELLGO — Design "Operations Console"

Paket desain buat WELLGO yang di-deploy di Streamlit. Identitas: **petroleum/instrument**
dengan satu *urgency spectrum* yang konsisten (warna = makna, bukan dekorasi).

## Isi

| File | Fungsi |
|---|---|
| `.streamlit/config.toml` | Theme native Streamlit (warna dasar, font) |
| `wellgo_ui.py` | Design system: `inject_theme()` + komponen (`hero_header`, `kpi_row`, `unit_card`, dst) |
| `wellgo_demo.py` | Demo siap-jalan dengan data dummy |

## Jalanin demo

```bash
pip install streamlit pandas
python -m streamlit run wellgo_demo.py
```

## Pasang ke app.py asli

```python
import streamlit as st
import wellgo_ui as ui

st.set_page_config(page_title="WELLGO", page_icon="🛢️", layout="wide")
ui.inject_theme()                          # panggil SEKALI, paling atas

# header — kasih angka real dari hasil plan_week
ui.hero_header(date_str=days[0].strftime("%d %b %Y"),
               horizon=len(days), units=9,
               compliance=int(100 * sched_ok / total), mode=mode)

# KPI — dari unit_summary / agregasi lo
ui.kpi_row([
    ("wells scheduled", f"{n_sched}", f"/{n_total}", ui.TEAL_GREEN),
    ("total route",     f"{total_km:.0f}", " km", ui.TEAL),
    ("avg utilization", f"{avg_util:.0f}", "%",  ui.AMBER),
])

# per hari + per unit (loop hasil plan_week)
for day_idx, day in enumerate(days, 1):
    rows = summary[summary.day_idx == day_idx]
    ui.day_header(f"Day {day_idx}", day.strftime("%a %d %b"),
                  units=rows.plan_unit.nunique(), wells=len(rows_wells))
    for _, u in rows.iterrows():
        ui.unit_card(u.Unit, u["Sub-area"], km=u["Rute (km)"],
                     minutes=u["Est (min)"], pct=u.budget_pct,
                     wells=[(w.well, (w.max_date - day).days,
                             "NW" if w.is_nwaws else None) for _, w in wells_of_unit])
```

## Catatan

- **Urgency spectrum** dipetakan langsung dari model (`ui.urgency_color(days, flag)`):
  on-time → watch → critical → overdue, + violet buat NW/AWS. Konsisten dipakai
  di chip deadline, load-gauge, dan legend.
- Font (Space Grotesk / Inter / JetBrains Mono) di-load dari Google Fonts via CSS.
  Kalau deploy di env tanpa internet, host font sendiri & ganti `@import`-nya.
- Selektor CSS nge-target `data-testid` Streamlit yang relatif stabil. Kalau
  upgrade Streamlit dan ada yang geser (tabs/metric/sidebar), tinggal sesuaikan di
  `inject_theme()`. Diuji di Streamlit ≥ 1.30.
- `config.toml` set `toolbarMode="minimal"` — hapus kalau butuh menu deploy default.
