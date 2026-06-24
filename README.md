# Well Test Grouping Optimizer — 9 Unit MWT (H-1 Daily Planner)

Grouping kandidat pengetesan sumur untuk **9 unit tes (MWT / MPAS_xxx)**, tiap unit
handle **5-6 sumur/hari** dari territory-nya sendiri.

## Cara jalanin (WAJIB lewat terminal, BUKAN tombol Run)

```bash
cd "C:\Users\feriyanto\Downloads\Grouping Well"
python -m pip install streamlit pandas numpy openpyxl pydeck   # sekali aja
python -m streamlit run app.py
```
Browser kebuka di http://localhost:8501. Jangan jalanin `python app.py`.

## Logika

- **Filter**: hanya unit MPAS (TS di-exclude). Eligible = `min_date ≤ target ≤ max_date`,
  belum dieksekusi, koordinat ada.
- **Per unit**: tiap MPAS_xxx pilih `N` sumur paling mendesak (deadline terdekat) dari
  territory-nya, lalu urut rute nearest-neighbor. Sisanya = backlog.
- **Time budget** (opsional): kalau rute N sumur paling urgent > budget, buang sumur paling
  tidak urgent sampai muat. Default 360 menit, bisa di-uncheck biar count (5-6) yg dominan.
- **AT-RISK**: sumur deadline ≤1 hari yg gak kebagian slot ditandai merah — risiko lewat
  compliance, perlu tindakan (tambah shift / re-prioritas).
- **Backlog & carry-over**: sumur yg belum dieksekusi otomatis balik jadi kandidat besok.

## Realita kapasitas

9 unit × 5-6 = **45-54 sumur/hari**, padahal eligible bisa 300+. Jadi sebagian besar
backlog; sistem mastiin yg paling mendesak duluan & nge-flag yg berisiko telat.

## New well tanpa koordinat

- Coordinate cache (SQLite): input lat/lon manual sekali → kesimpen permanen.
- Imputasi best-effort dari centroid field/subarea kalau ada sibling bercoordinat.
- Panel "Tanpa koordinat" buat input manual sumur baru.

## Next step

- Engine seleksi greedy + nearest-neighbor. Mau rute optimal (TSP/CVRP exact) → ganti
  `plan_units` / `nn_route` pakai Google OR-Tools, UI gak perlu diubah.
- Bisa di-integrate ke SIELO / WELL-STARLING sebagai modul scheduling.
