"""
wellgo_ui.py — Design system "Operations Console" untuk WELLGO (Streamlit).
"""
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PALET — petroleum / instrument
# ─────────────────────────────────────────────────────────────────────────────
INK        = "#0B2027"
INK_2      = "#102E37"
INK_3      = "#0F252D"
TEAL       = "#15A3A3"
TEAL_DEEP  = "#0E7C7C"
TEAL_BRIGHT= "#2BC4C4"
SURFACE    = "#EEF2F3"
CARD       = "#FFFFFF"
BORDER     = "#DCE4E6"
MUTED      = "#5E7076"
MUTED_HEAD = "#8DA3A9"

# Header — biru dongker (navy)
HEADER_BG    = "#000080"   # navy murni (background header)
HEADER_CHIP  = "#2A2A94"   # chip status di header (navy lebih terang)
HEADER_STRIP = "#00006E"   # strip legend kategori (navy lebih gelap)

# Status compliance — warna sinyal, dituning biar kontras tinggi di header navy
HD_GOOD = "#34D399"   # >= 100%  on-track  (emerald terang)
HD_WARN = "#FBBF24"   # 90–99%   watch     (amber terang)
HD_CRIT = "#F87171"   # < 90%    kritis    (coral terang)

TEAL_GREEN = "#1F9D72"   # RTN (Routine)
AMBER      = "#E6B23A"   # AWS (Annual Well Status)
ORANGE     = "#E67E22"   # Peringatan General / NCMP
BLUE       = "#3B82F6"   # PRQ (PE Request)
RED        = "#D6473A"   # ORQ (Ops Request) / Kritis
VIOLET     = "#6B4FD8"   # NW (New Well)

_CAT_TINT = {
    "RTN": ("#E4F4EC", "#14674B"),
    "AWS": ("#FBF0D8", "#8A6410"),
    "PRQ": ("#E0F2FE", "#1E3A8A"),
    "ORQ": ("#FBE2DF", "#8A271E"),
    "NW":  ("#ECE7FB", "#3D2B8A"),
}

AREA_TINT = {
    "BANGKO":  ("#E7EEF6", "#234A78"),
    "BALAM":   ("#F3E9F4", "#6A2C6E"),
    "BEKASAP": ("#E4F4EC", "#14674B"),
    "LIBO":    ("#FBF0D8", "#8A6410"),
}

# Logo "well connecting" — Mark A (hub menghubungkan node sumur). Pakai currentColor.
LOGO_SVG = (
    '<svg width="22" height="22" viewBox="0 0 32 32" fill="none" '
    'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
    'stroke-linejoin="round"><line x1="16" y1="13.2" x2="16" y2="8.6"/>'
    '<line x1="13.6" y1="17.5" x2="8.7" y2="20.6"/>'
    '<line x1="18.4" y1="17.5" x2="23.3" y2="20.6"/>'
    '<circle cx="16" cy="6" r="2.6"/><circle cx="6.5" cy="22" r="2.6"/>'
    '<circle cx="25.5" cy="22" r="2.6"/>'
    '<circle cx="16" cy="16" r="2.8" fill="currentColor" stroke="none"/></svg>'
)

def inject_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{ --wg-ink:{INK}; --wg-teal:{TEAL}; --wg-surface:{SURFACE}; --wg-card:{CARD}; --wg-border:{BORDER}; --wg-muted:{MUTED}; }}
    html, body, [class*="css"], .stApp {{ font-family:'Inter',sans-serif; color:{INK}; }}
    .stApp {{ background:{SURFACE}; }}
    .wg-mono, [data-testid="stMetricValue"] {{ font-family:'Inter',sans-serif; font-variant-numeric:tabular-nums; letter-spacing:-0.02em; }}
    h1,h2,h3,h4 {{ font-family:'Inter',sans-serif; font-weight:600; letter-spacing:-0.01em; color:{INK}; }}

    #MainMenu, footer {{ visibility:hidden; height:0; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    .block-container {{ padding-top:3rem; padding-bottom:2rem; max-width:1180px; }}

    .stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ font-family:'Inter',sans-serif; font-weight:500; font-size:14px; color:{MUTED}; padding:8px 16px; background:transparent; border-radius:8px 8px 0 0; }}
    .stTabs [aria-selected="true"] {{ color:{INK} !important; background:{CARD}; border:1px solid {BORDER}; border-bottom:2px solid {TEAL}; }}

    .stButton > button {{ font-family:'Inter',sans-serif; font-weight:500; border-radius:9px; border:1px solid {BORDER}; background:{CARD}; color:{INK}; transition:all .12s ease; }}
    .stButton > button:hover {{ border-color:{TEAL}; color:{TEAL_DEEP}; }}
    .stButton > button[kind="primary"] {{ background:{TEAL}; color:#fff; border-color:{TEAL}; }}
    .stButton > button[kind="primary"]:hover {{ background:{TEAL_DEEP}; color:#fff; }}

    [data-testid="stSidebar"] {{ background-color:#F8FAFC !important; border-right:1px solid {BORDER}; }}
    [data-testid="stSidebar"] .block-container {{ padding-top:4rem; }}
    [data-testid="stSidebar"] div[data-baseweb="select"] > div, [data-testid="stSidebar"] input[type="text"], [data-testid="stSidebar"] input[type="number"], [data-testid="stSidebar"] div[data-baseweb="input"] {{ background-color: #FFFFFF !important; border-radius: 6px; }}

    [data-testid="stMetric"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:12px 14px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05); }}
    [data-testid="stMetricLabel"] p {{ font-family:'Inter',sans-serif; font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#7C8E94; font-weight:600; }}
    [data-testid="stSlider"] [role="slider"] {{ background:{TEAL}; }}
    [data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:10px; }}

    .wg-eyb {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#7C8E94; font-weight: 600; font-family:'Inter',sans-serif; }}
    .wg-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05); }}
    .wg-chip {{ font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:500; padding:4px 8px; border-radius:6px; display:inline-flex; gap:5px; align-items:center; border: 1px solid rgba(0,0,0,0.05); }}
    .wg-disp {{ font-family:'Inter',sans-serif; }}
    </style>
    """, unsafe_allow_html=True)

def hero_header(date_str, horizon, units, compliance, mode="pooled"):
    comp_color = HD_GOOD if compliance >= 100 else (HD_WARN if compliance >= 90 else HD_CRIT)
    chips = "".join(f'<span class="wg-chip" style="background:{HEADER_CHIP};color:#FFFFFF;font-family:\'Inter\',sans-serif;">{t}</span>' for t in [date_str, f"horizon {horizon}d", f"{units} units active", f"{mode} mode"])
    legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:6px;font-size:11px;font-family:\'Inter\',sans-serif;color:#FFFFFF;font-weight:500;"><span style="width:10px;height:10px;border-radius:3px;background:{c};"></span>{lbl}</span>' for c, lbl in [(VIOLET, "NW"), (AMBER, "AWS"), (TEAL_GREEN, "RTN"), (BLUE, "PRQ"), (RED, "ORQ")])
    st.markdown(f"""
    <div style="border-radius:13px;overflow:hidden;margin-bottom:14px;border:1px solid {BORDER};box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
      <div style="background:{HEADER_BG};padding:16px 20px 14px;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <span class="wg-disp" style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border:1.5px solid {TEAL};border-radius:9px;color:{TEAL_BRIGHT};">{LOGO_SVG}</span>
            <div>
              <div class="wg-disp" style="font-size:21px;font-weight:600;color:#FFFFFF;letter-spacing:-0.01em;line-height:1.1;">WELLGO</div>
              <div style="font-size:12px;color:rgba(255,255,255,0.75);font-weight:400;margin-top:2px;">Well Grouping Optimizer · SL North · PT Pertamina Hulu Rokan </div>
            </div>
          </div>
          <div style="text-align:right;">
            <div class="wg-mono" style="font-size:24px;font-weight:700;color:{comp_color};line-height:1;">{compliance}%</div>
            <div class="wg-eyb" style="color:rgba(255,255,255,0.75);margin-top:4px;">on-deadline</div>
          </div>
        </div>
        <div style="display:flex;gap:7px;margin-top:13px;flex-wrap:wrap;">{chips}</div>
      </div>
      <div style="background:{HEADER_STRIP};padding:10px 20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <span class="wg-eyb" style="color:rgba(255,255,255,0.75);">Category</span>{legend}
      </div>
    </div>
    """, unsafe_allow_html=True)

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, unit, accent) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="wg-card" style="padding:11px 13px;position:relative;overflow:hidden;">
              <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{accent};"></div>
              <div class="wg-eyb">{label}</div>
              <div class="wg-mono" style="font-size:24px;font-weight:600;margin-top:4px;color:{INK};">
                {value}<span style="color:#9AA9AE;font-size:14px;font-weight:500;">{unit}</span></div>
            </div>""", unsafe_allow_html=True)

def day_header(day_label, date_str, units, wells):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:16px 0 12px;">
      <span class="wg-disp" style="font-size:16px;font-weight:600;color:{INK};">{day_label}</span>
      <span style="font-size:13px;color:{MUTED};font-weight:500;">· {date_str}</span>
      <span style="flex:1;height:1px;background:{BORDER};margin: 0 8px;"></span>
      <span class="wg-mono" style="font-size:12px;color:{MUTED};font-weight:500;background:#F8FAFC;padding:2px 8px;border-radius:12px;border:1px solid {BORDER};">{units} units · {wells} wells</span>
    </div>""", unsafe_allow_html=True)

def unit_card(unit, area, km, minutes, pct, wells):
    """
    Format wells param = list of (well_id, category, window_string, duration_string)
    """
    a_bg, a_tx = AREA_TINT.get(str(area).upper(), ("#EAEFF0", INK))
    bar = min(max(pct, 0), 100)
    bar_color = RED if pct > 100 else (AMBER if pct > 90 else TEAL)

    chips = ""
    for w in wells:
        wid, cat, win_str, dur_str = w
        bg, tx = _CAT_TINT.get(cat, ("#F1F5F9", "#334155"))
        chips += (f'<span class="wg-chip" style="background:{bg};color:{tx};">'
                  f'{wid} <span style="opacity:0.4;margin:0 2px;">|</span> '
                  f'<span style="font-size:10px;font-family:\'Inter\',sans-serif;">{win_str} <span style="opacity:0.5;">•</span> {dur_str}</span></span>')

    st.markdown(f"""
    <div class="wg-card" style="padding:16px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;">
          <span class="wg-disp" style="font-size:16px;font-weight:600;color:{INK};">{unit}</span>
          <span class="wg-chip" style="background:{a_bg};color:{a_tx};font-family:'Inter',sans-serif;font-size:10px;border:none;">{area}</span>
        </div>
        <div style="display:flex;gap:16px;background:#F8FAFC;padding:4px 10px;border-radius:8px;border:1px solid {BORDER};">
          <span><span class="wg-mono" style="font-size:14px;font-weight:600;color:{INK};">{km:.0f}</span>
            <span style="font-size:11px;color:#64748B;font-weight:500;"> km</span></span>
          <span><span class="wg-mono" style="font-size:14px;font-weight:600;color:{INK};">{minutes:.0f}</span>
            <span style="font-size:11px;color:#64748B;font-weight:500;"> min</span></span>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
        <div style="flex:1;height:6px;background:#E2E8F0;border-radius:4px;overflow:hidden;">
          <div style="width:{bar}%;height:100%;background:{bar_color};border-radius:4px;"></div></div>
        <span class="wg-mono" style="font-size:12px;color:{MUTED};font-weight:500;">{pct:.0f}% budget</span>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;">{chips}</div>
    </div>""", unsafe_allow_html=True)

def section(title, eyebrow=None):
    eyb = f'<div class="wg-eyb" style="margin-bottom:4px;">{eyebrow}</div>' if eyebrow else ""
    st.markdown(f'<div style="margin:20px 0 12px;">{eyb}'
                f'<div class="wg-disp" style="font-size:17px;font-weight:600;letter-spacing:-0.01em;color:{INK};">{title}</div></div>',
                unsafe_allow_html=True)
