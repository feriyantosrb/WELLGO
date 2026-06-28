"""
wellgo_guide.py — Halaman "Panduan" in-app untuk WELLGO.

Pakai di app.py:
    import wellgo_guide as guide
    ...
    tabs = st.tabs(["Schedule", "Map", "Units", "Analysis", "Panduan"])
    with tabs[-1]:
        guide.render_guide()

Atau jalankan standalone buat preview:
    python -m streamlit run wellgo_guide.py
"""
import streamlit as st
import wellgo_ui as ui

# kategori: (kode, warna, arti)
_CATS = [
    ("NW",  ui.VIOLET,     "New Well — sumur baru, prioritas absolut (menerobos antrean)."),
    ("AWS", ui.AMBER,      "After Well Service — prioritas tinggi."),
    ("RTN", ui.TEAL_GREEN, "Routine — pengetesan rutin reguler."),
    ("PRQ", ui.BLUE,       "PE Request — permintaan dari Petroleum Engineer."),
    ("ORQ", ui.RED,        "Ops Request — permintaan operasi, umumnya kritis."),
]

_GLOSS = [
    ("MWT", "Mobile Well Test — kru pengujian sumur bergerak (9 unit)."),
    ("MPAS", "Penamaan unit/kru MWT."),
    ("TS", "Test Station — fasilitas pengujian sumur di Test Station."),
    ("GP / GAS", "Sumur bertipe gas (ditandai label [GAS])."),
    ("COMP", "Completed — sumur sudah selesai dites."),
    ("NCMP", "Not Completed — sumur gagal/belum selesai dites."),
    ("OFF", "Sumur mati / tidak berproduksi."),
    ("Horizon", "Rentang hari yang direncanakan dalam satu siklus."),
    ("Elastic Limit", "Batas persebaran rute (km) untuk mencegah chaining."),
    ("On-Deadline", "Persentase sumur yang dijadwalkan dalam tenggatnya."),
]


def _sec(num, title):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin:22px 0 12px;'
        f'border-bottom:2px solid {ui.TEAL};padding-bottom:8px;">'
        f'<span style="width:30px;height:30px;background:{ui.HEADER_BG};color:#fff;'
        f'border-radius:7px;font-weight:700;display:inline-flex;align-items:center;'
        f'justify-content:center;font-size:15px;">{num}</span>'
        f'<span style="font-size:18px;font-weight:700;color:{ui.HEADER_BG};">{title}</span>'
        f'</div>', unsafe_allow_html=True)


def _callout(kind, title, body):
    bg, bar, tx = {
        "tip":  ("#E4F4EC", ui.TEAL_GREEN, "#14674B"),
        "warn": ("#FBF0D8", ui.AMBER,      "#8A6410"),
    }[kind]
    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {bar};border-radius:8px;'
        f'padding:12px 16px;margin:8px 0;">'
        f'<div style="font-weight:700;color:{tx};font-size:11px;letter-spacing:.5px;'
        f'text-transform:uppercase;margin-bottom:3px;">{title}</div>'
        f'<div style="font-size:13.5px;color:#2A3B42;">{body}</div></div>',
        unsafe_allow_html=True)


def _card(html):
    st.markdown(f'<div class="wg-card" style="padding:14px 18px;margin-bottom:10px;">{html}</div>',
                unsafe_allow_html=True)


def render_guide(version="1.0", area="Sumatra Light North (SLN)"):
    # ── Hero ────────────────────────────────────────────────────────────────
    chips = "".join(
        f'<span style="background:{ui.HEADER_CHIP};color:#fff;font-size:12px;'
        f'padding:4px 12px;border-radius:6px;margin-right:7px;">{t}</span>'
        for t in [f"Versi {version}", f"Area: {area}"])
    st.markdown(
        f'<div style="background:{ui.HEADER_BG};border-radius:13px;padding:22px 24px;'
        f'margin-bottom:8px;display:flex;align-items:center;gap:16px;">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:48px;height:48px;border:1.5px solid {ui.TEAL};border-radius:11px;'
        f'color:{ui.TEAL_BRIGHT};">{ui.LOGO_SVG}</span>'
        f'<div><div style="font-size:22px;font-weight:700;color:#fff;letter-spacing:-.01em;">'
        f'Panduan Penggunaan</div>'
        f'<div style="font-size:13px;color:{ui.TEAL_BRIGHT};margin:2px 0 8px;">WELLGO · Well Grouping Optimizer</div>'
        f'<div>{chips}</div></div></div>', unsafe_allow_html=True)
    st.caption("Platform optimasi berbasis algoritma spasial untuk optimasi penjadwalan dan rute "
               "9 kru Mobile Well Test (MWT) dengan menyeimbangkan jarak, durasi pengetesan, dan tenggat waktu sumur.")

    # ── 1. Tentang ──────────────────────────────────────────────────────────
    _sec(1, "Tentang WELLGO")
    st.markdown(
        "WELLGO (Well Grouping Optimizer) adalah platform berbasis algoritma optimasi spasial yang dikembangkan untuk mengoptimalkan penjadwalan dan rute kru Mobile Well Test (MWT) di area Sumatra Light North. Sistem ini secara simultan menyeimbangkan jarak tempuh, durasi pengujian, dan tenggat waktu tiap sumur guna memastikan efisiensi operasional serta mencegah terlewatnya target pengujian.")

    # ── 2. Quick Start ──────────────────────────────────────────────────────
    _sec(2, "Alur Kerja Cepat")
    st.markdown("Ikuti 4 langkah berurutan dari menu **Sidebar** di sebelah kiri:")
    steps = [
        ("Upload Data", "Masukkan master Excel “Kandidat Sumur” dan “Data Spasial” (dalam 1 file Excel)."),
        ("Filter Area", "Singkirkan area yang tidak beroperasi atau sedang down."),
        ("Upload Realisasi (opsional)", "Masukkan file COMP/NCMP harian untuk menghilangkan sumur yang sudah selesai dari antrean."),
        ("Tentukan Horizon & Jalankan", "Pilih tanggal siklus, atur parameter kelenturan, lalu klik **Re-run Optimizer**."),
    ]
    for i, (t, b) in enumerate(steps, 1):
        st.markdown(
            f'<div style="display:flex;gap:10px;margin-bottom:7px;align-items:flex-start;">'
            f'<span style="flex:none;width:22px;height:22px;background:{ui.TEAL};color:#fff;'
            f'border-radius:50%;font-size:12px;font-weight:700;display:inline-flex;'
            f'align-items:center;justify-content:center;">{i}</span>'
            f'<div style="font-size:14px;"><b>{t}.</b> {b}</div></div>',
            unsafe_allow_html=True)

    # ── 3. Sidebar ──────────────────────────────────────────────────────────
    _sec(3, "Konfigurasi Sidebar")
    _card("<b>3.1 &nbsp;Manajemen Data & Filter</b>"
          "<ul style='margin:6px 0 0 18px;font-size:13.5px;'>"
          "<li><b>Sheet Kandidat & Spasial</b>: nama sheet harus sama persis dengan di file Excel.</li>"
          "<li><b>Hanya Unit Tes (MPAS)</b>: centang → sumur TS dialihkan ke MWT (saat fasilitas TS down).</li>"
          "<li><b>Exclude Area</b>: kecualikan area dari perhitungan (default: LIBO).</li></ul>")
    _card("<b>3.2 &nbsp;Status Realisasi Harian</b>"
          "<ul style='margin:6px 0 0 18px;font-size:13.5px;'>"
          "<li>Upload laporan eksekusi harian (<b>SCHDatabase</b>).</li>"
          "<li><b>Skip Sumur NCMP berstatus OFF</b>: sumur yang gagal dites dan saat ini statusnya mati (OFF) tidak dijadwalkan ulang.</li></ul>")
    _card("<b>3.3 &nbsp;Horizon Perencanaan</b>"
          "<ul style='margin:6px 0 0 18px;font-size:13.5px;'>"
          "<li><b>Rentang Siklus</b>: batas total kalender minggu ini.</li>"
          "<li><b>Mulai Planning</b>: titik awal algoritma mengisi keranjang unit MWT.</li></ul>")
    _card("<b>3.4 &nbsp;Parameter Algoritma</b> <span style='color:#5E7076;'>(dalam kotak Form)</span>"
          "<ul style='margin:6px 0 0 18px;font-size:13.5px;'>"
          "<li><b>Target Sumur / Unit</b>: kapasitas maksimal per kru per hari (default: 6).</li>"
          "<li><b>Batas Persebaran Rute (Elastic Limit)</b>: set 5 km → sistem tak merangkai sumur yang jarak ujung-ke-ujung > 5 km.</li>"
          "<li><b>Kecepatan Rata-rata</b>: konversi jarak (km) → estimasi waktu (menit).</li></ul>")
    _callout("warn", "Penting", "Tombol <b>Re-run Optimizer</b> harus diklik setiap kali Anda mengubah parameter di kotak Form agar jadwal diperbarui.")

    # ── 4. Workspace ────────────────────────────────────────────────────────
    _sec(4, "Membaca Layar Utama (Workspace)")
    st.markdown("Layar utama: deretan **Indikator KPI** di atas, dan **6 Tab Detail** di bawahnya.")
    k1, k2, k3 = st.columns(3)
    for col, (lbl, desc) in zip(
        [k1, k2, k3],
        [("On-Deadline (%)", "Persentase sumur yang terjadwal dan tidak melewati deadline."),
         ("Miss Deadline", "Sumur kritis yang gagal kebagian kru MWT karena kapasitas penuh setelah dilakukan optimasi."),
         ("Wells OFF", "Sumur mati (OFF) yang di-skip dari perencanaan.")]):
        with col:
            st.markdown(
                f'<div class="wg-card" style="padding:12px 14px;border-top:3px solid {ui.TEAL};">'
                f'<div class="wg-eyb">{lbl}</div>'
                f'<div style="font-size:13px;margin-top:4px;color:#324;">{desc}</div></div>',
                unsafe_allow_html=True)

    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
    st.markdown("**Legenda Kategori Sumur** — warna ini konsisten di seluruh aplikasi:")
    chips = "".join(
        f'<span style="background:{c};color:#fff;font-size:12px;font-weight:700;'
        f'padding:3px 11px;border-radius:5px;margin:0 5px 5px 0;display:inline-block;">{code}</span>'
        for code, c, _ in _CATS)
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
    rows = "".join(
        f'<tr><td style="width:70px;font-weight:700;color:{ui.HEADER_BG};padding:5px 4px;'
        f'border-bottom:1px solid #EEF2F3;">{code}</td>'
        f'<td style="padding:5px 4px;border-bottom:1px solid #EEF2F3;font-size:13.5px;">{desc}</td></tr>'
        for code, _, desc in _CATS)
    st.markdown(f'<table style="width:100%;border-collapse:collapse;margin-top:6px;">{rows}</table>',
                unsafe_allow_html=True)

    # ── 5. Tabs ─────────────────────────────────────────────────────────────
    _sec(5, "Enam Tab Detail")
    tabs_info = [
        ("TAB 1", "Jadwal Operasional",
         ["Rincian sumur per hari, dipisah <b>Remote</b> (Bangko/Balam) dan <b>Non-Remote</b> (Bekasap).",
          "Tiap kartu unit menampilkan jarak, % pemakaian jam kerja (budget), label <b>[GAS]</b> jika tipe GP.",
          "<b>Atur Manual (Add/Remove):</b> keluarkan sumur paksa, atau tarik sumur sisa ke unit tertentu."]),
        ("TAB 2", "Peta Rute",
         ["Visualisasi rute MWT; filter warna per <b>Unit</b> atau status (Early/Late/On-time).",
          "<b>Fokus Tanggal:</b> lihat blok poligon jangkauan area tiap unit.",
          "<b>Pencarian Cepat:</b> ketik nama sumur (mis. BO083) → peta zoom-in & menandai lokasi."]),
        ("TAB 3", "Matriks Deviasi",
         ["Evaluasi sumur yang dites mendahului (<b>EARLY</b>) atau telat (<b>LATE</b>).",
          "Sumur NW dan AWS dikunci sistem (prioritas absolut) → tidak memicu Early/Late."]),
        ("TAB 4", "Cart Manual",
         ["<b>Matriks Ketersediaan:</b> slot kru mana yang masih hijau dan mana yang Over.",
          "<b>Smart Cart Assistant:</b> sarankan unit terdekat untuk sumur Miss Deadline."]),
        ("TAB 5", "SCH Database",
         ["Mini-dashboard yang membedah file laporan harian.",
          "Rekap sumur COMP, NCMP yang dijadwal ulang, dan NCMP yang di-skip karena OFF."]),
        ("TAB 6", "Diagnostik",
         ["Tempat menyimpan pekerjaan.",
          "<b>Simpan Status Eksekusi:</b> tandai sumur yang benar-benar dikerjakan hari itu.",
          "<b>Unduh Rute (.xlsx):</b> ekspor hasil optimasi ke Excel untuk operator lapangan."]),
    ]
    for tn, tt, items in tabs_info:
        lis = "".join(f"<li>{x}</li>" for x in items)
        st.markdown(
            f'<div class="wg-card" style="padding:0;margin-bottom:10px;overflow:hidden;">'
            f'<div style="background:{ui.SURFACE};padding:9px 14px;border-bottom:1px solid {ui.BORDER};">'
            f'<span style="background:{ui.HEADER_BG};color:#fff;font-size:11px;font-weight:700;'
            f'padding:2px 8px;border-radius:4px;margin-right:10px;">{tn}</span>'
            f'<b style="color:{ui.HEADER_BG};">{tt}</b></div>'
            f'<div style="padding:10px 14px;"><ul style="margin:0 0 0 18px;font-size:13.5px;">{lis}</ul></div>'
            f'</div>', unsafe_allow_html=True)

    # ── 6. Tips ─────────────────────────────────────────────────────────────
    _sec(6, "Tips & Troubleshooting")
    _callout("tip", "Atasi “Chaining” Jarak",
             "Jika satu unit rutenya memanjang puluhan km padahal jarak antar sumurnya pendek, "
             "<b>turunkan Elastic Limit</b> (mis. ke 4 km) lalu klik <b>Re-run</b>.")
    _callout("tip", "Prioritas NW / AWS",
             "Sumur <b>New Well</b> (ungu) dan <b>AWS</b> (orange) selalu menerobos antrean. "
             "sistem pantang memindahkan jadwal NW keluar dari batas Min → Max Date-nya.")
    _callout("warn", "Hard Reset",
             "Jika Assign Manual menumpuk dan rute berantakan, scroll ke bawah halaman dan klik "
             "<b>Hard Reset Konfigurasi</b> untuk kembali ke perhitungan murni (factory reset).")

    # ── 7. Glossary ─────────────────────────────────────────────────────────
    _sec(7, "Glosarium Istilah")
    rows = "".join(
        f'<tr><td style="width:110px;font-weight:700;color:{ui.HEADER_BG};padding:6px 4px;'
        f'border-bottom:1px solid #EEF2F3;vertical-align:top;">{k}</td>'
        f'<td style="padding:6px 4px;border-bottom:1px solid #EEF2F3;font-size:13.5px;">{v}</td></tr>'
        for k, v in _GLOSS)
    st.markdown(f'<table style="width:100%;border-collapse:collapse;">{rows}</table>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(page_title="WELLGO — Panduan", page_icon="wellgo_icon.png", layout="wide")
    ui.inject_theme()
    render_guide()
