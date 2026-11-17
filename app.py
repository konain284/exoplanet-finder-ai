import streamlit as st
import numpy as np
import lightkurve as lk
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Exoplanet Finder AI", page_icon="🪐", layout="wide",
                   initial_sidebar_state="collapsed")

AMBER, CYAN, INK, MUTE, LINE, PANEL = "#ffb454", "#5ad1e6", "#e8ecf4", "#8794ab", "#1a2538", "#0b111c"

# ───────────────────────── Styling ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{ --bg:#05070d; --panel:#0b111c; --line:#1a2538; --ink:#e8ecf4; --mute:#8794ab; --amber:#ffb454; --cyan:#5ad1e6; }
html, body, [class*="css"], .stApp { font-family:'Space Grotesk',sans-serif; color:var(--ink); }
.stApp{
  background:
    radial-gradient(1px 1px at 12% 18%, #fff9 50%, transparent 51%),
    radial-gradient(1px 1px at 78% 12%, #fff7 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 41% 33%, #fffc 50%, transparent 51%),
    radial-gradient(1px 1px at 90% 48%, #fff6 50%, transparent 51%),
    radial-gradient(1px 1px at 22% 62%, #fff8 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 64% 74%, #fff9 50%, transparent 51%),
    radial-gradient(ellipse at 50% -10%, #12203a 0%, var(--bg) 60%);
  background-attachment: fixed;
}
header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ max-width:1240px; padding-top:2rem; padding-bottom:4rem; }
.hero{ display:grid; grid-template-columns:1.1fr 1fr; gap:2rem; align-items:center; margin-bottom:1.5rem; }
.hero h1{ font-size:3.1rem; font-weight:700; letter-spacing:-0.03em; line-height:1.05; margin:0 0 .8rem; }
.hero p{ color:var(--mute); font-size:1.05rem; line-height:1.6; max-width:34rem; margin:0; }
.badges{ display:flex; gap:.5rem; margin-top:1.2rem; flex-wrap:wrap; }
.badge{ font-family:'IBM Plex Mono',monospace; font-size:.75rem; color:var(--cyan);
  border:1px solid var(--line); background:#0b111c99; padding:.3rem .7rem; border-radius:999px; }
.planet{ animation: cross 9s linear infinite; }
.curve{ stroke-dasharray:1; stroke-dashoffset:1; animation: draw 9s linear infinite; }
@keyframes cross{ 0%{transform:translateX(-170px);opacity:1} 85%{transform:translateX(170px);opacity:1} 95%,100%{transform:translateX(170px);opacity:0} }
@keyframes draw{ 0%{stroke-dashoffset:1;opacity:1} 85%{stroke-dashoffset:0;opacity:1} 95%,100%{stroke-dashoffset:0;opacity:0} }
@media (prefers-reduced-motion: reduce){ .planet,.curve{ animation:none; } .curve{ stroke-dashoffset:0; } }
@media (max-width: 900px){ .hero{ grid-template-columns:1fr; } .hero h1{ font-size:2.3rem; } }
.panel-title{ font-size:.95rem; font-weight:600; margin:0 0 .2rem; }
.panel-sub{ font-size:.82rem; color:var(--mute); margin:0 0 1rem; }
div[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--panel); border:1px solid var(--line) !important; border-radius:14px; }
label, .stTextInput label p, .stNumberInput label p, .stSlider label p, .stToggle label p{ color:var(--mute) !important; font-size:.82rem !important; }
div[data-baseweb="input"], div[data-baseweb="base-input"]{ background:#070c15 !important; border-radius:10px !important; border:1px solid var(--line) !important; }
div[data-baseweb="input"]:focus-within{ border-color:var(--cyan) !important; box-shadow:0 0 0 1px var(--cyan); }
input{ color:var(--ink) !important; font-family:'IBM Plex Mono',monospace !important; }
.stButton > button{ width:100%; background:var(--amber); color:#1a1205; border:none; border-radius:10px;
  padding:.65rem 1.2rem; font-weight:600; transition:filter .2s, transform .2s; }
.stButton > button:hover{ filter:brightness(1.1); transform:translateY(-1px); color:#1a1205; }
.preset .stButton > button{ background:transparent; color:var(--cyan); border:1px solid var(--line); font-weight:500; padding:.4rem .6rem; font-size:.8rem; }
.preset .stButton > button:hover{ border-color:var(--cyan); color:var(--cyan); }
.result{ display:flex; gap:1.6rem; align-items:center; }
.verdict-title{ font-size:1.5rem; font-weight:600; letter-spacing:-0.01em; margin:0 0 .3rem; }
.verdict-text{ color:var(--mute); font-size:.92rem; line-height:1.55; margin:0; }
.readouts{ display:grid; grid-template-columns:repeat(4,1fr); margin-top:1.2rem; border-top:1px solid var(--line); }
.ro{ padding:.9rem 1rem .2rem 0; }
.ro + .ro{ padding-left:1rem; border-left:1px solid var(--line); }
.ro .k{ color:var(--mute); font-size:.74rem; }
.ro .v{ font-family:'IBM Plex Mono',monospace; font-size:1.15rem; margin-top:.15rem; }
.empty{ color:var(--mute); font-size:.95rem; line-height:1.6; padding:3rem 1rem; text-align:center; }
@media (max-width: 700px){ .readouts{ grid-template-columns:repeat(2,1fr); } .result{ flex-direction:column; align-items:flex-start; } }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Hero ─────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <h1>Exoplanet Finder AI</h1>
    <p>Pick a star, let a 1D ResNet read its light curve from NASA MAST, then probe the model: see what it looked at, replay the transit, and stress-test it with noise.</p>
    <div class="badges"><span class="badge">NASA Kepler</span><span class="badge">MAST archive</span><span class="badge">1D ResNet + XAI</span></div>
  </div>
  <svg viewBox="0 0 600 220" role="img" style="width:100%;height:auto">
    <defs><radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="55%" stop-color="#ffb454" stop-opacity=".55"/><stop offset="100%" stop-color="#ffb454" stop-opacity="0"/>
    </radialGradient></defs>
    <circle cx="300" cy="70" r="78" fill="url(#glow)"/><circle cx="300" cy="70" r="46" fill="#ffc87a"/>
    <circle class="planet" cx="300" cy="70" r="11" fill="#05070d"/>
    <line x1="40" y1="170" x2="560" y2="170" stroke="#1a2538" stroke-dasharray="3 5"/>
    <path class="curve" pathLength="1" d="M40 170 L214 170 L228 183 L372 183 L386 170 L560 170" fill="none" stroke="#5ad1e6" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    <text x="40" y="205" fill="#8794ab" font-size="12" font-family="IBM Plex Mono, monospace">brightness over time</text>
  </svg>
</div>
""", unsafe_allow_html=True)

# ───────────────────────── Model ─────────────────────────
class AdvancedExoplanetResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.res_conv1 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.res_conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 50 + 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_flux, x_meta):
        x = self.pool(self.relu(self.conv1(x_flux)))
        residual = x
        out = self.relu(self.res_conv1(x))
        out = self.res_conv2(out)
        x = self.relu(out + residual)
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, x_meta), dim=1)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

@st.cache_resource
def load_model():
    m = AdvancedExoplanetResNet()
    m.load_state_dict(torch.load("exoplanet_model.pth", map_location=torch.device("cpu")))
    m.eval()
    return m

model = load_model()

# ───────────────────────── Core functions ─────────────────────────
def predict_batch(flux_batch, period, snr):
    """flux_batch: (N, 201) normalized flux -> (N,) probabilities."""
    x = torch.tensor(np.asarray(flux_batch), dtype=torch.float32).unsqueeze(1)
    meta = torch.tensor([[period / 10.0, min(snr / 50.0, 1.0)]], dtype=torch.float32).repeat(x.shape[0], 1)
    with torch.no_grad():
        return model(x, meta).squeeze(1).numpy()

def occlusion_map(flux, period, snr, w=15):
    """Real XAI: hide a window of the curve and measure how much the planet score drops."""
    base = predict_batch(flux[None], period, snr)[0]
    batch = np.repeat(flux[None], len(flux), axis=0)
    for i in range(len(flux)):
        batch[i, max(0, i - w // 2): i + w // 2 + 1] = 0.0
    drop = np.clip(base - predict_batch(batch, period, snr), 0, None)
    return drop / (drop.max() + 1e-9)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_and_fold(kic, period, t0):
    sr = lk.search_lightcurve(f"KIC {kic}", mission="Kepler")
    if len(sr) == 0:
        raise ValueError("No Kepler light curve found for this KIC ID.")
    lc = sr[0].download(quality_bitmask="default")
    if lc is None:
        raise ValueError("Could not download the light curve.")
    folded = lc.flatten(window_length=401).fold(period=period, epoch_time=t0)
    binned = folded.bin(bins=201)
    bfl = np.nan_to_num(np.array(binned.flux.value, dtype=np.float32), nan=1.0)
    bph = np.array(binned.phase.value, dtype=np.float64)
    if len(bfl) != 201:
        bfl = np.interp(np.linspace(0, 1, 201), np.linspace(0, 1, len(bfl)), bfl).astype(np.float32)
        bph = np.interp(np.linspace(0, 1, 201), np.linspace(0, 1, len(bph)), bph)
    norm = (bfl - bfl.mean()) / (bfl.std() + 1e-8)
    return dict(ph=np.array(folded.phase.value, dtype=np.float64), fl=np.array(folded.flux.value, dtype=np.float64),
                bph=bph, bfl=bfl.astype(np.float64), norm=norm.astype(np.float32), n=len(folded.flux))

def base_layout(fig, h=420):
    fig.update_layout(height=h, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTE, family="Space Grotesk"),
                      margin=dict(l=10, r=10, t=30, b=10), hoverlabel=dict(bgcolor="#070c15", font_color=INK),
                      legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig

def gauge(p, thr):
    circ = 2 * np.pi * 54
    color = AMBER if p > thr else CYAN
    return f"""<svg width="150" height="150" viewBox="0 0 150 150">
      <circle cx="75" cy="75" r="54" fill="none" stroke="#1a2538" stroke-width="10"/>
      <circle cx="75" cy="75" r="54" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"
        stroke-dasharray="{circ:.1f}" stroke-dashoffset="{circ*(1-p):.1f}" transform="rotate(-90 75 75)"/>
      <text x="75" y="80" text-anchor="middle" fill="#e8ecf4" font-size="28" font-weight="500" font-family="IBM Plex Mono, monospace">{p*100:.1f}%</text>
      <text x="75" y="100" text-anchor="middle" fill="#8794ab" font-size="10" font-family="Space Grotesk, sans-serif">planet probability</text></svg>"""

# ───────────────────────── State & presets ─────────────────────────
PRESETS = {  # approximate values from literature, BKJD epochs
    "Kepler-1b": ("11446443", 2.4706, 122.76),
    "Kepler-7b": ("5780885", 4.8855, 134.28),
    "Kepler-8b": ("6922244", 3.5225, 121.11),
    "Kepler-10b": ("11904151", 0.8375, 131.57),
}
for k, v in dict(kic="2304235", period=0.57, t0=133.15, snr=15.0).items():
    st.session_state.setdefault(k, v)

def apply_preset(name):
    st.session_state.kic, st.session_state.period, st.session_state.t0 = PRESETS[name]

# ───────────────────────── Tabs ─────────────────────────
tab1, tab2, tab3 = st.tabs(["🔭 Exoplanet Analyzer", "🧪 Stress-Test Lab", "📊 Model Intelligence"])

with tab1:
    left, right = st.columns([1, 2.3], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<p class="panel-title">Target</p><p class="panel-sub">Choose a known planet host or enter your own.</p>', unsafe_allow_html=True)
            st.markdown('<div class="preset">', unsafe_allow_html=True)
            pc = st.columns(2)
            for i, name in enumerate(PRESETS):
                pc[i % 2].button(name, key=f"p_{name}", on_click=apply_preset, args=(name,))
            st.markdown('</div>', unsafe_allow_html=True)
            st.text_input("Kepler ID (KIC)", key="kic")
            st.number_input("Orbital period (days)", key="period", format="%.4f")
            st.number_input("Transit epoch (BKJD)", key="t0", format="%.4f")
            st.number_input("Model SNR", key="snr", format="%.2f")
            analyze = st.button("Analyze light curve")
        with st.container(border=True):
            st.markdown('<p class="panel-title">Decision threshold</p><p class="panel-sub">Move it and watch the verdict change.</p>', unsafe_allow_html=True)
            thr = st.slider("Planet if probability above", 0.05, 0.95, 0.50, 0.05)

    if analyze:
        with st.status("Running analysis…", expanded=True) as status:
            try:
                st.write("📡 Fetching light curve from NASA MAST")
                d = fetch_and_fold(st.session_state.kic, float(st.session_state.period), float(st.session_state.t0))
                st.write("🧠 Running 1D ResNet")
                p = float(predict_batch(d["norm"][None], st.session_state.period, st.session_state.snr)[0])
                st.write("🔍 Computing occlusion map (XAI)")
                d["xai"] = occlusion_map(d["norm"], st.session_state.period, st.session_state.snr)
                d.update(p=p, kic=st.session_state.kic, period=float(st.session_state.period), snr=float(st.session_state.snr))
                med, mn = np.nanmedian(d["bfl"]), np.nanmin(d["bfl"])
                d["depth"] = max(0.0, (med - mn) * 1e6)
                st.session_state.res = d
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(f"{e}")

    with right:
        res = st.session_state.get("res")
        if not res:
            with st.container(border=True):
                st.markdown('<div class="empty">No analysis yet.<br>Pick a preset or enter a star, then select <b>Analyze light curve</b>.</div>', unsafe_allow_html=True)
        else:
            p, period = res["p"], res["period"]
            is_planet = p > thr
            title = "Planet candidate detected" if is_planet else "Likely not a planet"
            text = ("The 1D ResNet found a coherent transit dip that matches planetary characteristics."
                    if is_planet else "The signal lacks a consistent transit depth or features of a valid planet profile.")
            with st.container(border=True):
                st.markdown(f"""
                <div class="result">{gauge(p, thr)}<div>
                  <p class="verdict-title" style="color:{AMBER if is_planet else INK}">{title}</p>
                  <p class="verdict-text">{text}<br>Threshold: {thr:.0%}</p></div></div>
                <div class="readouts">
                  <div class="ro"><div class="k">Star</div><div class="v">KIC {res['kic']}</div></div>
                  <div class="ro"><div class="k">Period</div><div class="v">{period:.4f} d</div></div>
                  <div class="ro"><div class="k">Transit depth</div><div class="v">{res['depth']:,.0f} ppm</div></div>
                  <div class="ro"><div class="k">Observations</div><div class="v">{res['n']:,}</div></div>
                </div>""", unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<p class="panel-title">Folded light curve & what the model looked at</p><p class="panel-sub">Zoom, pan and hover. The bottom strip shows how much the score drops when that part of the curve is hidden.</p>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                show_raw = c1.toggle("Raw observations", True)
                show_xai = c2.toggle("XAI importance", True)
                zoom = c3.toggle("Zoom on transit", False)

                rows = 2 if show_xai else 1
                fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                    row_heights=[0.75, 0.25] if show_xai else [1])
                if show_raw:
                    fig.add_trace(go.Scattergl(x=res["ph"], y=res["fl"], mode="markers", name="Observations",
                                  marker=dict(size=3, color=CYAN, opacity=0.3)), row=1, col=1)
                fig.add_trace(go.Scatter(x=res["bph"], y=res["bfl"], mode="lines+markers", name="Binned trend",
                              line=dict(color=AMBER, width=2.5), marker=dict(size=4)), row=1, col=1)
                if show_xai:
                    fig.add_trace(go.Bar(x=res["bph"], y=res["xai"], name="Importance",
                                  marker=dict(color=res["xai"], colorscale=[[0, "#12203a"], [1, AMBER]]),
                                  hovertemplate="importance %{y:.2f}<extra></extra>"), row=2, col=1)
                    fig.update_yaxes(title_text="importance", row=2, col=1)
                fig.update_yaxes(title_text="Normalized flux", row=1, col=1)
                fig.update_xaxes(title_text="Time from transit centre (days)", row=rows, col=1)
                if zoom:
                    w = 0.15 * period
                    fig.update_xaxes(range=[-w, w])
                    m = (res["bph"] >= -w) & (res["bph"] <= w)
                    if m.any():
                        lo, hi = np.nanmin(res["bfl"][m]), np.nanmax(res["bfl"][m])
                        pad = (hi - lo) * 0.6 + 1e-5
                        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
                st.plotly_chart(base_layout(fig, 480), use_container_width=True)

            with st.container(border=True):
                st.markdown('<p class="panel-title">Transit replay</p><p class="panel-sub">Drag the slider to move the planet across its star and see the matching point on the curve.</p>', unsafe_allow_html=True)
                bph, bfl = res["bph"], res["bfl"]
                span = float(np.nanmax(np.abs(bph))) or 0.5 * period
                ph = st.slider("Orbital phase (days from mid-transit)", -span, span, -span, span / 100, format="%.3f")
                med, mn = np.nanmedian(bfl), np.nanmin(bfl)
                dip = bph[bfl < med - 0.5 * (med - mn)] if med > mn else np.array([])
                half = max((dip.max() - dip.min()) / 2, span * 0.02) if len(dip) > 1 else span * 0.1
                rp = float(np.clip(np.sqrt(max(med - mn, 0) / max(med, 1e-9)), 0.07, 0.4))
                px = ph / half
                cur = np.interp(ph, bph[np.isfinite(bph)], bfl[np.isfinite(bph)])
                fig2 = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], horizontal_spacing=0.06)
                for r, a in ((1.35, 0.08), (1.15, 0.15), (1.0, 1.0)):
                    fig2.add_shape(type="circle", x0=-r, x1=r, y0=-r, y1=r, xref="x", yref="y", line_width=0,
                                   fillcolor="#ffc87a", opacity=a)
                fig2.add_shape(type="circle", x0=px - rp, x1=px + rp, y0=-rp, y1=rp, xref="x", yref="y",
                               fillcolor="#05070d", line=dict(color="#05070d"))
                fig2.add_trace(go.Scatter(x=bph, y=bfl, mode="lines", line=dict(color=CYAN, width=2), showlegend=False), row=1, col=2)
                fig2.add_trace(go.Scatter(x=[ph], y=[cur], mode="markers", showlegend=False,
                                          marker=dict(size=14, color=AMBER, line=dict(color="#fff", width=1))), row=1, col=2)
                fig2.update_xaxes(range=[-3, 3], visible=False, row=1, col=1)
                fig2.update_yaxes(range=[-1.5, 1.5], visible=False, scaleanchor="x", row=1, col=1)
                fig2.update_xaxes(title_text="Phase (days)", row=1, col=2)
                st.plotly_chart(base_layout(fig2, 330), use_container_width=True)

            st.download_button("⬇ Download binned curve + XAI (CSV)",
                               "phase_days,flux,importance\n" + "\n".join(
                                   f"{a:.6f},{b:.6f},{c:.4f}" for a, b, c in zip(res["bph"], res["bfl"], res["xai"])),
                               file_name=f"KIC{res['kic']}_analysis.csv", mime="text/csv")

with tab2:
    res = st.session_state.get("res")
    st.markdown("### 🧪 Stress-Test Lab")
    st.caption("Does the model hold up when the data gets worse? Run an analysis first, then break it on purpose.")
    if not res:
        with st.container(border=True):
            st.markdown('<div class="empty">Analyze a star in the first tab to unlock the lab.</div>', unsafe_allow_html=True)
    else:
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                st.markdown('<p class="panel-title">Noise injection</p><p class="panel-sub">Add random noise to the folded curve and re-score it live.</p>', unsafe_allow_html=True)
                sigma = st.slider("Noise level (σ)", 0.0, 3.0, 0.0, 0.1)
                seed = st.number_input("Random seed", value=7, step=1)
                noisy = res["norm"] + np.random.default_rng(int(seed)).normal(0, sigma, 201).astype(np.float32)
                pn = float(predict_batch(noisy[None], res["period"], res["snr"])[0])
                st.metric("Planet probability", f"{pn*100:.1f}%", f"{(pn - res['p'])*100:+.1f} pts vs clean")
                f = go.Figure()
                f.add_trace(go.Scatter(x=res["bph"], y=res["norm"], name="Clean", line=dict(color=CYAN, width=2)))
                f.add_trace(go.Scatter(x=res["bph"], y=noisy, name="Noisy", line=dict(color=AMBER, width=1.5)))
                st.plotly_chart(base_layout(f, 320), use_container_width=True)
        with b:
            with st.container(border=True):
                st.markdown('<p class="panel-title">SNR sensitivity</p><p class="panel-sub">The model also receives SNR as input. See how much it leans on it.</p>', unsafe_allow_html=True)
                snrs = np.linspace(0, 50, 26)
                probs = [float(predict_batch(res["norm"][None], res["period"], s)[0]) for s in snrs]
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=snrs, y=probs, mode="lines+markers", name="Probability", line=dict(color=AMBER, width=2.5)))
                f2.add_hline(y=thr, line_dash="dash", line_color=MUTE, annotation_text="threshold")
                f2.add_trace(go.Scatter(x=[res["snr"]], y=[res["p"]], mode="markers", name="Your input",
                                        marker=dict(size=13, color=CYAN, line=dict(color="#fff", width=1))))
                f2.update_yaxes(range=[0, 1], title_text="Planet probability")
                f2.update_xaxes(title_text="SNR")
                st.plotly_chart(base_layout(f2, 400), use_container_width=True)

with tab3:
    st.markdown("### 📊 Model Training & Evaluation Insights")
    st.write("How the 1D ResNet + metadata model was trained on NASA Kepler TCE data.")
    m1, m2 = st.columns(2, gap="large")
    with m1:
        with st.container(border=True):
            st.markdown('<p class="panel-title">Training loss (10 epochs)</p><p class="panel-sub">Loss fell from 0.687 to 0.147.</p>', unsafe_allow_html=True)
            losses = [0.6874, 0.5222, 0.4549, 0.4429, 0.3571, 0.2956, 0.2286, 0.2202, 0.1493, 0.1467]
            f3 = go.Figure(go.Scatter(x=list(range(1, 11)), y=losses, mode="lines+markers", name="BCELoss",
                                      line=dict(color=AMBER, width=2.5), marker=dict(size=8),
                                      fill="tozeroy", fillcolor="rgba(255,180,84,0.08)"))
            f3.update_xaxes(title_text="Epoch", dtick=1)
            f3.update_yaxes(title_text="Loss")
            st.plotly_chart(base_layout(f3, 330), use_container_width=True)
    with m2:
        with st.container(border=True):
            st.markdown('<p class="panel-title">Architecture</p><p class="panel-sub">Pick a stage to see its shape.</p>', unsafe_allow_html=True)
            stages = {
                "Input · flux + metadata": "201 flux bins (1 channel) and 2 metadata values: period/10 and SNR/50.",
                "Conv1d + MaxPool": "16 filters, kernel 5, then pool by 2: 201 → 100 bins.",
                "Residual block": "Two Conv1d layers (16 filters, kernel 3) with a skip connection.",
                "Conv1d + MaxPool (2)": "32 filters, kernel 5, then pool by 2: 100 → 50 bins.",
                "Dense head": "1,600 conv features + 2 metadata → 64 units → sigmoid probability.",
            }
            pick = st.radio("Stage", list(stages), label_visibility="collapsed")
            st.info(stages[pick])
            st.markdown("**Loss:** Binary cross-entropy · **Optimizer:** Adam (lr 0.001) · **Data:** NASA Exoplanet Archive, Kepler DR25 TCE")