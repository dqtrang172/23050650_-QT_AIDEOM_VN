from __future__ import annotations

from pathlib import Path
from itertools import product
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="VN AIDEOM-VN | Decision Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root{
  --bg:#0B1020; --panel:#111A2E; --panel2:#17233A; --text:#F5F7FB;
  --muted:#9AA7BD; --line:#26344F; --pink:#FF4B77; --cyan:#35D4D4;
  --green:#22C55E; --yellow:#F6C453; --blue:#62A8FF;
}
html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;}
.stApp {background: radial-gradient(circle at 12% 0%, #17223A 0%, #0B1020 42%, #080C18 100%); color:var(--text);}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #172235 0%, #0E1627 100%); border-right:1px solid var(--line);}
section[data-testid="stSidebar"] * {color:#EAF0FF;}
.block-container {padding-top:2.2rem; padding-bottom:3rem; max-width:1500px;}
h1, h2, h3 {letter-spacing:-0.03em;}
h1 {font-size:2.55rem !important; font-weight:860 !important; margin-bottom:0.3rem !important;}
h2 {font-size:1.65rem !important; font-weight:780 !important; margin-top:1.4rem !important;}
h3 {font-size:1.15rem !important; font-weight:740 !important;}
hr {border-color:var(--line); margin:1.2rem 0;}
.kicker {display:inline-flex; gap:.45rem; align-items:center; font-size:.78rem; font-weight:750; color:#08111f; background:linear-gradient(90deg,#41F2B3,#35D4D4); padding:.35rem .65rem; border-radius:999px; text-transform:uppercase; letter-spacing:.04em;}
.subtle {color:var(--muted); font-size:0.98rem; line-height:1.55;}
.card {background:linear-gradient(180deg, rgba(25,35,58,.88), rgba(14,22,39,.88)); border:1px solid var(--line); border-radius:18px; padding:18px 18px; box-shadow:0 10px 28px rgba(0,0,0,.26);}
.metric-card {background:linear-gradient(135deg, rgba(255,75,119,.18), rgba(53,212,212,.06)); border:1px solid rgba(255,255,255,.08); border-radius:18px; padding:17px 18px; min-height:126px;}
.metric-label {font-size:.84rem; color:#AAB5C7; font-weight:650; margin-bottom:.25rem;}
.metric-value {font-size:2rem; color:#FF4B77; font-weight:850; letter-spacing:-.04em;}
.metric-delta {display:inline-flex; background:rgba(34,197,94,.14); color:#52E397; border:1px solid rgba(34,197,94,.25); padding:.15rem .45rem; border-radius:999px; font-size:.78rem; font-weight:700; margin-top:.35rem;}
.agent {background:linear-gradient(135deg, rgba(98,168,255,.16), rgba(255,75,119,.08)); border:1px solid rgba(98,168,255,.24); border-radius:18px; padding:17px 18px; margin-top:18px;}
.agent-title {font-weight:850; color:#F9FBFF; font-size:1.05rem; margin-bottom:.5rem;}
.agent ul {margin:.25rem 0 .1rem 1.1rem; padding:0;}
.agent li {margin:.35rem 0; color:#D9E2F2; line-height:1.48;}
.badge {display:inline-block; padding:.22rem .55rem; border-radius:999px; background:#1C2945; border:1px solid #2A3A5C; color:#DDE7FB; font-size:.78rem; font-weight:700; margin-right:.35rem;}
.small-note {font-size:.82rem; color:#91A0B9;}
.stTabs [data-baseweb="tab-list"] {gap:.3rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"] {background:#121B2F; border:1px solid #23314D; border-radius:13px 13px 0 0; color:#DCE7FA; padding:.55rem .9rem;}
.stTabs [aria-selected="true"] {background:linear-gradient(135deg,#FF4B77,#7C5CFF) !important; color:white !important; border-color:transparent !important;}
[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:14px; overflow:hidden;}
.stButton>button {background:linear-gradient(90deg,#FF4B77,#7C5CFF); color:white; border:0; border-radius:12px; padding:.55rem .8rem; font-weight:800;}
.stDownloadButton>button {background:#192743; color:#EAF0FF; border:1px solid #2B3D62; border-radius:12px;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATA & HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    macro = pd.read_csv(DATA_DIR / "vietnam_macro_2020_2025.csv")
    sectors = pd.read_csv(DATA_DIR / "vietnam_sectors_2024.csv")
    regions = pd.read_csv(DATA_DIR / "vietnam_regions_2024.csv")
    return macro, sectors, regions

macro, sectors, regions = load_data()


def fmt(x, digits=2):
    try:
        if abs(float(x)) >= 1000:
            return f"{float(x):,.{digits}f}"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def section_title(kicker: str, title: str, desc: str | None = None):
    st.markdown(f"<span class='kicker'>{kicker}</span>", unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if desc:
        st.markdown(f"<div class='subtle'>{desc}</div>", unsafe_allow_html=True)
    st.markdown("---")


def metric_card(label: str, value: str, delta: str | None = None):
    html = f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>"
    if delta:
        html += f"<div class='metric-delta'>{delta}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def agent_box(title: str, bullets: list[str]):
    items = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(
        f"""
<div class="agent">
  <div class="agent-title">🤖 Tác nhân phân tích kết quả — {title}</div>
  <ul>{items}</ul>
</div>
""",
        unsafe_allow_html=True,
    )


def plot_bar(df, x, y, title, orientation="v", text=None):
    fig = px.bar(df, x=x, y=y, text=text, orientation=orientation, title=title, template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF0FF"), title=dict(font=dict(size=18)),
        margin=dict(l=10, r=10, t=55, b=20), height=430,
    )
    fig.update_traces(marker_line_width=0, textposition="outside")
    return fig


def plot_line(df, x, y, title, markers=True):
    fig = px.line(df, x=x, y=y, markers=markers, title=title, template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF0FF"), title=dict(font=dict(size=18)),
        margin=dict(l=10, r=10, t=55, b=20), height=420,
    )
    fig.update_traces(line=dict(width=3))
    return fig


def downloadable_csv(df: pd.DataFrame, filename: str):
    st.download_button(
        "⬇️ Tải bảng CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def minmax_good(s: pd.Series):
    denom = s.max() - s.min()
    return (s - s.min()) / denom if denom != 0 else s * 0


def minmax_bad(s: pd.Series):
    denom = s.max() - s.min()
    return (s.max() - s) / denom if denom != 0 else s * 0


# -----------------------------------------------------------------------------
# BÀI 1
# -----------------------------------------------------------------------------
def cobb_data(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08):
    theta = max(0.0, 1 - alpha - beta - gamma - delta)
    df = macro[["year", "GDP_trillion_VND", "digital_economy_share_GDP_pct"]].copy()
    # Theo bảng trong đề bài. CSV macro không chứa đủ K, L, AI, H nên ghép thêm các chuỗi được cung cấp trong đề.
    df["K"] = [16500, 17800, 19600, 21300, 23500, 25900]
    df["L"] = [53.6, 50.5, 51.7, 52.4, 52.9, 53.4]
    df["D"] = [12.0, 12.7, 14.3, 16.5, 18.3, 19.5]
    df["AI"] = [55.6, 60.2, 65.4, 67.0, 73.8, 80.1]
    df["H"] = [24.1, 26.1, 26.2, 27.0, 28.4, 29.2]
    Y = df["GDP_trillion_VND"].to_numpy(float)
    K = df["K"].to_numpy(float)
    L = df["L"].to_numpy(float)
    D = df["D"].to_numpy(float)
    AI = df["AI"].to_numpy(float)
    H = df["H"].to_numpy(float)
    A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    A_bar = A.mean()
    Y_hat = A_bar * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100
    df["TFP_A_t"] = A
    df["Y_hat"] = Y_hat
    df["APE_pct"] = np.abs((Y - Y_hat) / Y) * 100
    return df, theta, mape, A_bar


def page_bai1():
    section_title("Cấp độ dễ", "Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng", "Tính TFP, kiểm tra sai số dự báo, phân rã tăng trưởng và mô phỏng GDP 2030.")
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    with c1: alpha = st.slider("α — Vốn K", 0.20, 0.45, 0.33, 0.01)
    with c2: beta = st.slider("β — Lao động L", 0.25, 0.55, 0.42, 0.01)
    with c3: gamma = st.slider("γ — Số hóa D", 0.03, 0.20, 0.10, 0.01)
    with c4: delta = st.slider("δ — AI", 0.02, 0.16, 0.08, 0.01)
    theta = max(0, 1-alpha-beta-gamma-delta)
    with c5: metric_card("θ — Nhân lực H tự động", fmt(theta,2), "CRS = 1")

    df, theta, mape, A_bar = cobb_data(alpha, beta, gamma, delta)
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("TFP trung bình", fmt(A_bar, 4))
    with k2: metric_card("MAPE dự báo", f"{mape:.2f}%")
    with k3: metric_card("GDP thực tế 2025", f"{macro.loc[macro.year==2025,'GDP_billion_USD'].iloc[0]:,.1f} tỷ USD")
    with k4: metric_card("Kinh tế số/GDP 2025", f"{macro.loc[macro.year==2025,'digital_economy_share_GDP_pct'].iloc[0]:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["TFP A_t", "Dự báo & MAPE", "Growth accounting", "Dự báo 2030"])
    with tab1:
        st.dataframe(df[["year","GDP_trillion_VND","K","L","D","AI","H","TFP_A_t"]], use_container_width=True, hide_index=True)
        st.plotly_chart(plot_line(df, "year", "TFP_A_t", "TFP A_t calibrated theo năm"), use_container_width=True)
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.year, y=df.GDP_trillion_VND, mode="lines+markers", name="Y thực tế"))
        fig.add_trace(go.Scatter(x=df.year, y=df.Y_hat, mode="lines+markers", name="Ŷ dự báo"))
        fig.update_layout(template="plotly_dark", title="So sánh GDP thực tế và dự báo", height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[["year","GDP_trillion_VND","Y_hat","APE_pct"]], use_container_width=True, hide_index=True)
    with tab3:
        contrib = []
        cols = {"K": alpha, "L": beta, "D": gamma, "AI": delta, "H": theta}
        for col, w in cols.items():
            contrib.append([col, np.log(df[col].iloc[-1]/df[col].iloc[0]) * w])
        total_y = np.log(df.GDP_trillion_VND.iloc[-1]/df.GDP_trillion_VND.iloc[0])
        tfp = np.log(df.TFP_A_t.iloc[-1]/df.TFP_A_t.iloc[0])
        contrib.append(["TFP", tfp])
        cdf = pd.DataFrame(contrib, columns=["Yếu tố", "Đóng góp log growth"])
        cdf["Tỷ trọng trong tăng trưởng (%)"] = cdf["Đóng góp log growth"] / total_y * 100
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        st.plotly_chart(plot_bar(cdf, "Yếu tố", "Tỷ trọng trong tăng trưởng (%)", "Phân rã tăng trưởng 2020-2025", text="Tỷ trọng trong tăng trưởng (%)"), use_container_width=True)
    with tab4:
        d2030 = st.slider("D — Kinh tế số/GDP 2030 (%)", 22.0, 40.0, 30.0, 0.5)
        ai2030 = st.slider("AI — nghìn DN số 2030", 85.0, 140.0, 100.0, 1.0)
        h2030 = st.slider("H — lao động qua đào tạo 2030 (%)", 30.0, 45.0, 35.0, 0.5)
        gK = st.slider("Tăng K bình quân/năm", 0.02, 0.10, 0.06, 0.005)
        gL = st.slider("Tăng L bình quân/năm", 0.00, 0.04, 0.006, 0.001)
        A2030 = df.TFP_A_t.iloc[-1] * (1 + 0.012)**5
        K2030 = df.K.iloc[-1] * (1+gK)**5
        L2030 = df.L.iloc[-1] * (1+gL)**5
        Y2030 = A2030 * K2030**alpha * L2030**beta * d2030**gamma * ai2030**delta * h2030**theta
        metric_card("GDP dự báo 2030", f"{Y2030:,.1f} nghìn tỷ VND", "kịch bản người dùng chỉnh")
    agent_box("Bài 1", [
        f"MAPE hiện tại là {mape:.2f}%, đủ thấp để dùng như mô hình minh họa nhưng không nên xem là dự báo chính thức.",
        "Nếu tăng γ và δ, vai trò của số hóa/AI trong phân rã tăng trưởng tăng rõ, nhưng θ giảm vì ràng buộc CRS.",
        "Kịch bản 2030 hợp lý nhất là tăng D và H đồng thời; chỉ tăng AI mà thiếu nhân lực số sẽ tạo kết quả thiếu bền vững."
    ])


# -----------------------------------------------------------------------------
# BÀI 2
# -----------------------------------------------------------------------------
def solve_budget_lp(B=100, h_min=20, strategic=0.35):
    c = -np.array([0.85, 1.20, 0.95, 1.35])
    A = [
        [1,1,1,1], [-1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,-1],
        [strategic, -1+strategic, strategic, -1+strategic]
    ]
    b = [B, -25, -15, -h_min, -10, 0]
    return linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0,None)]*4, method="highs")


def page_bai2():
    section_title("Cấp độ dễ", "Bài 2 — LP phân bổ ngân sách số", "Tối đa hóa GDP gain kỳ vọng với 4 hạng mục đầu tư số và ràng buộc chính sách.")
    c1,c2,c3 = st.columns(3)
    with c1: B = st.slider("Ngân sách tổng B (nghìn tỷ)", 80, 160, 100, 5)
    with c2: hmin = st.slider("Sàn nhân lực số x₃", 10, 45, 20, 1)
    with c3: strategic = st.slider("Tỷ trọng AI + R&D tối thiểu", 0.20, 0.55, 0.35, 0.01)
    res = solve_budget_lp(B, hmin, strategic)
    labels = ["Hạ tầng số", "AI & dữ liệu", "Nhân lực số", "R&D công nghệ"]
    if not res.success:
        st.error("Bài toán không khả thi với bộ tham số hiện tại. Hãy giảm sàn hoặc tăng ngân sách.")
        return
    alloc = pd.DataFrame({"Hạng mục": labels, "Phân bổ": res.x, "Hệ số tác động": [0.85,1.20,0.95,1.35]})
    alloc["GDP gain"] = alloc["Phân bổ"] * alloc["Hệ số tác động"]
    k1,k2,k3 = st.columns(3)
    with k1: metric_card("Z* tối ưu", f"{-res.fun:,.2f}")
    with k2: metric_card("Tổng phân bổ", f"{res.x.sum():,.1f}")
    with k3: metric_card("AI + R&D", f"{(res.x[1]+res.x[3])/res.x.sum()*100:.1f}%")
    st.dataframe(alloc, use_container_width=True, hide_index=True)
    st.plotly_chart(plot_bar(alloc, "Hạng mục", "Phân bổ", "Cơ cấu phân bổ ngân sách tối ưu", text="Phân bổ"), use_container_width=True)

    sens = []
    for bb in [100, 120, 140, B]:
        r = solve_budget_lp(bb, hmin, strategic)
        sens.append([bb, -r.fun if r.success else np.nan])
    sdf = pd.DataFrame(sens, columns=["Ngân sách", "Z* tối ưu"]).drop_duplicates("Ngân sách").sort_values("Ngân sách")
    st.plotly_chart(plot_line(sdf, "Ngân sách", "Z* tối ưu", "Độ nhạy Z*(B)"), use_container_width=True)
    agent_box("Bài 2", [
        "R&D thường nhận phần dư lớn vì hệ số tác động cao nhất, nhưng vẫn phải giữ sàn cho hạ tầng, AI và nhân lực.",
        f"Với B={B}, nghiệm dùng {res.x.sum():.1f}/{B} nghìn tỷ; nếu ngân sách tăng, Z* tăng gần tuyến tính trong vùng ràng buộc chưa đổi.",
        "Shadow price của ngân sách có thể hiểu gần đúng là mức GDP gain tăng thêm khi nới 1 đơn vị ngân sách, nhưng cần kiểm tra theo từng đoạn độ nhạy."
    ])


# -----------------------------------------------------------------------------
# BÀI 3
# -----------------------------------------------------------------------------
def priority_table(weights):
    df = sectors.copy()
    df["productivity_proxy"] = df["gdp_share_2024_pct"] / df["labor_share_pct"].replace(0, np.nan)
    cols = ["growth_rate_2024_pct", "productivity_proxy", "spillover_coef_0_1", "export_billion_USD", "labor_million", "ai_readiness_0_100"]
    norm = df[cols].apply(minmax_good)
    risk_good = minmax_bad(df["automation_risk_pct"])
    W = np.array(weights[:6])
    wrisk = weights[6]
    # Risk đã đảo thành chỉ số tốt, nên cộng. Nhãn vẫn thể hiện giảm rủi ro.
    df["Priority"] = norm.values @ W + wrisk * risk_good.values
    for c in cols:
        df[f"norm_{c}"] = norm[c]
    df["norm_low_risk"] = risk_good
    return df.sort_values("Priority", ascending=False)


def page_bai3():
    section_title("Cấp độ dễ", "Bài 3 — Chỉ số ưu tiên 10 ngành", "Chuẩn hóa min-max, tính Priorityᵢ và kiểm tra độ nhạy trọng số AI readiness.")
    st.markdown("<span class='badge'>Trọng số mặc định</span> growth 0.15 · productivity 0.15 · spillover 0.20 · export 0.15 · employment 0.10 · AI 0.20 · low risk 0.15", unsafe_allow_html=True)
    ai_w = st.slider("Trọng số AI readiness", 0.05, 0.40, 0.20, 0.01)
    base_other = np.array([0.15,0.15,0.20,0.15,0.10,0.15])
    other = base_other / base_other.sum() * (1-ai_w)
    weights = [other[0], other[1], other[2], other[3], other[4], ai_w, other[5]]
    df = priority_table(weights)
    top = df.head(3)[["sector_name_vi","Priority"]]
    c1,c2,c3 = st.columns(3)
    for col, (_, row) in zip([c1,c2,c3], top.iterrows()):
        with col: metric_card(row["sector_name_vi"], f"{row['Priority']:.3f}", "Top ngành")
    st.dataframe(df[["sector_name_vi","Priority","growth_rate_2024_pct","ai_readiness_0_100","automation_risk_pct"]], use_container_width=True, hide_index=True)
    fig = plot_bar(df.sort_values("Priority"), "Priority", "sector_name_vi", "Xếp hạng ưu tiên ngành", orientation="h", text="Priority")
    st.plotly_chart(fig, use_container_width=True)

    heat = []
    for w_ai in np.arange(0.05, 0.401, 0.05):
        other = base_other / base_other.sum() * (1-w_ai)
        tmp = priority_table([other[0],other[1],other[2],other[3],other[4],w_ai,other[5]])
        for rank, (_, row) in enumerate(tmp.head(5).iterrows(), 1):
            heat.append({"AI_weight": round(w_ai,2), "Ngành": row["sector_name_vi"], "Rank": rank})
    hdf = pd.DataFrame(heat)
    st.dataframe(hdf.pivot_table(index="Ngành", columns="AI_weight", values="Rank", aggfunc="first"), use_container_width=True)
    agent_box("Bài 3", [
        f"Khi trọng số AI = {ai_w:.2f}, nhóm ưu tiên cao nhất là: {', '.join(top['sector_name_vi'].tolist())}.",
        "Ngành có AI readiness và lan tỏa cao thường giữ thứ hạng tốt; ngành có năng suất cao nhưng rủi ro/lượng việc làm thấp có thể tụt hạng.",
        "Nên dùng kết quả này như bảng gợi ý chính sách, không phải thứ tự tuyệt đối; trọng số cần được hội đồng chuyên gia thống nhất."
    ])


# -----------------------------------------------------------------------------
# BÀI 4
# -----------------------------------------------------------------------------
BETA = np.array([
    [1.15,0.85,0.55,1.30], [0.95,1.25,1.40,1.05], [1.05,0.95,0.85,1.15],
    [1.20,0.75,0.45,1.35], [0.90,1.30,1.55,1.00], [1.10,0.85,0.65,1.25]
])
REG_ABBR = ["TDMNPB", "ĐBSH", "BTB-DHMT", "Tây Nguyên", "ĐNB", "ĐBSCL"]
ITEMS = ["I", "D", "AI", "H"]


def solve_region_lp(budget=50000, floor=5000, cap=12000, hfloor=12000, fairness=True, lam=0.7):
    # 24 x variables + M
    n = 25
    c = np.zeros(n)
    c[:24] = -BETA.flatten()
    A, b = [], []
    A.append([1]*24 + [0]); b.append(budget)
    for r in range(6):
        row = np.zeros(n); row[r*4:(r+1)*4] = -1; A.append(row); b.append(-floor)
        row = np.zeros(n); row[r*4:(r+1)*4] = 1; A.append(row); b.append(cap)
    row = np.zeros(n); row[3:24:4] = -1; A.append(row); b.append(-hfloor)
    D0 = regions["digital_index_0_100"].to_numpy(float)
    gamma = 0.002
    if fairness:
        for r in range(6):
            row = np.zeros(n); row[r*4+1] = gamma; row[-1] = -1; A.append(row); b.append(-D0[r])
        for r in range(6):
            row = np.zeros(n); row[r*4+1] = -gamma; row[-1] = lam; A.append(row); b.append(D0[r])
    bounds = [(0,None)]*24 + [(0,None)]
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    return res


def page_bai4():
    section_title("Cấp độ trung bình", "Bài 4 — LP phân bổ ngân sách ngành-vùng", "Tối ưu 24 biến phân bổ theo 6 vùng và 4 hạng mục, có/không có ràng buộc công bằng vùng.")
    c1,c2,c3,c4 = st.columns(4)
    with c1: budget = st.slider("Ngân sách", 35000, 70000, 50000, 1000)
    with c2: floor = st.slider("Sàn mỗi vùng", 3000, 8000, 5000, 500)
    with c3: cap = st.slider("Trần mỗi vùng", 8000, 18000, 12000, 500)
    with c4: hfloor = st.slider("Sàn nhân lực H", 6000, 22000, 12000, 500)
    fairness = st.toggle("Bật ràng buộc công bằng vùng C5", value=True)
    res = solve_region_lp(budget, floor, cap, hfloor, fairness)
    if not res.success:
        st.error("Mô hình không khả thi. Hãy giảm sàn vùng/sàn H hoặc tăng ngân sách.")
        return
    X = res.x[:24].reshape(6,4)
    df = pd.DataFrame(X, columns=ITEMS, index=REG_ABBR).reset_index(names="Vùng")
    df["Tổng"] = df[ITEMS].sum(axis=1)
    metric_col = st.columns(3)
    with metric_col[0]: metric_card("Z* GDP gain", f"{-res.fun:,.0f}")
    with metric_col[1]: metric_card("Vùng nhận cao nhất", df.loc[df["Tổng"].idxmax(),"Vùng"])
    with metric_col[2]: metric_card("Hạng mục lớn nhất", ITEMS[int(np.argmax(X.sum(axis=0)))])
    st.dataframe(df, use_container_width=True, hide_index=True)
    fig = px.imshow(X, x=ITEMS, y=REG_ABBR, text_auto=".0f", template="plotly_dark", title="Heatmap phân bổ tối ưu")
    fig.update_layout(height=470, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    res_nf = solve_region_lp(budget, floor, cap, hfloor, False)
    if res_nf.success:
        cost = (-res_nf.fun) - (-res.fun)
        st.info(f"Chi phí kinh tế của ràng buộc công bằng vùng: khoảng **{cost:,.2f}** đơn vị GDP gain so với mô hình không C5.")
    agent_box("Bài 4", [
        "Đông Nam Bộ và Đồng bằng sông Hồng thường hút vốn AI/D do hệ số biên cao, còn vùng yếu cần ưu tiên H/I trước.",
        "Ràng buộc công bằng làm giảm Z* nhưng giúp tránh tập trung nguồn lực quá mức vào vùng đã sẵn sàng số.",
        "Kết quả heatmap là phần quan trọng nhất để giải thích chính sách: vùng nào nhận vốn, nhận vào hạng mục gì, vì sao."
    ])


# -----------------------------------------------------------------------------
# BÀI 5
# -----------------------------------------------------------------------------
PROJECTS = pd.DataFrame([
    [1,"Trung tâm dữ liệu quốc gia Hòa Lạc","Hạ tầng",12000,21500,8500,3500],
    [2,"Trung tâm dữ liệu quốc gia phía Nam","Hạ tầng",11500,20800,7500,4000],
    [3,"Hệ thống 5G phủ sóng toàn quốc","Hạ tầng",18000,32500,12000,6000],
    [4,"Hệ thống định danh điện tử VNeID 2.0","Chính phủ số",4500,9200,3500,1000],
    [5,"Cổng dịch vụ công quốc gia v3","Chính phủ số",3200,6800,2500,700],
    [6,"Y tế số quốc gia","Y tế số",5800,11400,4000,1800],
    [7,"Giáo dục số K-12 toàn quốc","Giáo dục",6500,12200,4500,2000],
    [8,"Trung tâm AI quốc gia + supercomputing","AI",15000,28500,9000,6000],
    [9,"Sandbox tài chính số","Tài chính số",2500,5800,1800,700],
    [10,"Logistics thông minh + cảng biển số","Logistics",7200,13800,5000,2200],
    [11,"Nông nghiệp số ĐBSCL","Nông nghiệp",4800,8500,3500,1300],
    [12,"Đào tạo 50.000 kỹ sư AI/bán dẫn","Nhân lực",8500,16200,5500,3000],
    [13,"Khu CN bán dẫn Bắc Ninh - Bắc Giang","Bán dẫn",20000,35000,13000,7000],
    [14,"An ninh mạng quốc gia (SOC)","An ninh",3800,7500,2800,1000],
    [15,"Open Data + dữ liệu mở quốc gia","Dữ liệu",1500,3800,1200,300],
], columns=["id","Tên dự án","Lĩnh vực","Chi phí","NPV","Năm 1-2","Năm 3-5"])


def mip_select(budget=80000, budget12=40000, force_both_dc=False, risk_adjust=False):
    best, bestv = None, -1
    probs = {"Hạ tầng":0.85,"Chính phủ số":0.75,"AI":0.65,"Bán dẫn":0.65,"Nhân lực":0.80,"Y tế số":0.80,"Giáo dục":0.80,"Tài chính số":0.80,"Logistics":0.80,"Nông nghiệp":0.80,"An ninh":0.80,"Dữ liệu":0.80}
    for bits in product([0,1], repeat=15):
        y = np.array(bits)
        n = y.sum()
        if n < 7 or n > 11: continue
        chosen = PROJECTS[y==1]
        if chosen["Chi phí"].sum() > budget or chosen["Năm 1-2"].sum() > budget12: continue
        if not force_both_dc and y[0] + y[1] > 1: continue
        if force_both_dc and not (y[0] == 1 and y[1] == 1): continue
        if y[7] > y[11]: continue  # P8 <= P12
        if y[12] > y[11]: continue # P13 <= P12
        if y[3] + y[4] < 1: continue
        if y[13] < 1: continue
        vals = PROJECTS["NPV"].to_numpy(float)
        if risk_adjust:
            vals = np.array([v*probs[lv] for v,lv in zip(PROJECTS["NPV"], PROJECTS["Lĩnh vực"])])
        val = (vals*y).sum()
        if val > bestv:
            bestv, best = val, y.copy()
    return best, bestv


def page_bai5():
    section_title("Cấp độ trung bình", "Bài 5 — MIP lựa chọn 15 dự án chuyển đổi số", "Duyệt tổ hợp nhị phân để giải knapsack tổng quát có ràng buộc loại trừ, tiên quyết và ngân sách đa năm.")
    c1,c2,c3 = st.columns(3)
    with c1: budget = st.slider("Ngân sách 5 năm", 60000, 110000, 80000, 5000)
    with c2: budget12 = st.slider("Ngân sách năm 1-2", 30000, 60000, 40000, 2500)
    with c3: risk_adj = st.toggle("Tối đa hóa NPV kỳ vọng theo rủi ro", value=False)
    force = st.toggle("Yêu cầu chọn cả P1 và P2", value=False)
    y, val = mip_select(budget, budget12, force, risk_adj)
    if y is None:
        st.error("Không tìm được tập dự án khả thi với cấu hình hiện tại.")
        return
    chosen = PROJECTS[y==1].copy()
    chosen["NPV/Chi phí"] = chosen["NPV"] / chosen["Chi phí"]
    k1,k2,k3,k4 = st.columns(4)
    with k1: metric_card("Số dự án chọn", str(int(y.sum())))
    with k2: metric_card("Tổng chi phí", f"{chosen['Chi phí'].sum():,.0f}")
    with k3: metric_card("Tổng lợi ích", f"{val:,.0f}")
    with k4: metric_card("NPV/Cost", f"{val/chosen['Chi phí'].sum():.2f}")
    st.dataframe(chosen, use_container_width=True, hide_index=True)
    st.plotly_chart(plot_bar(chosen.sort_values("NPV"), "NPV", "Tên dự án", "Lợi ích NPV các dự án được chọn", orientation="h", text="NPV"), use_container_width=True)
    y100, val100 = mip_select(100000, budget12, force, risk_adj)
    if y100 is not None:
        add = set(PROJECTS.loc[y100==1,"id"]) - set(PROJECTS.loc[y==1,"id"])
        st.info(f"Nếu nâng ngân sách 5 năm lên 100.000, lợi ích tối ưu là **{val100:,.0f}**. Dự án thêm mới: **{sorted(add) if add else 'không đổi'}**.")
    agent_box("Bài 5", [
        "MIP cho thấy dự án có tỷ suất cao chưa chắc được chọn nếu vướng ngân sách năm 1-2 hoặc ràng buộc tiên quyết.",
        "P14 an ninh mạng là ràng buộc bắt buộc; đây là ví dụ về mục tiêu an toàn không nên để mô hình tối đa hóa lợi ích loại bỏ.",
        "Khi bật NPV kỳ vọng theo rủi ro, nhóm AI/bán dẫn có thể giảm sức hút vì xác suất hoàn thành đúng tiến độ thấp hơn."
    ])


# -----------------------------------------------------------------------------
# BÀI 6
# -----------------------------------------------------------------------------
def topsis_scores(weights=None, entropy=False):
    criteria = ["grdp_per_capita_million_VND","fdi_registered_billion_USD","digital_index_0_100","ai_readiness_0_100","trained_labor_pct","rd_intensity_pct","internet_penetration_pct","gini_coef"]
    X = regions[criteria].to_numpy(float)
    if entropy:
        Xpos = X.copy()
        # chuyển Gini thành benefit trước khi entropy
        Xpos[:,-1] = Xpos[:,-1].max() - Xpos[:,-1] + 1e-6
        P = Xpos / Xpos.sum(axis=0)
        E = -(1/np.log(len(Xpos))) * np.sum(P*np.log(P+1e-12), axis=0)
        d = 1 - E
        w = d/d.sum()
    else:
        w = np.array(weights if weights is not None else [0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
    R = X / np.sqrt((X**2).sum(axis=0))
    V = R * w
    benefit = np.array([True,True,True,True,True,True,True,False])
    A_star = np.where(benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V-A_star)**2).sum(axis=1))
    S_neg = np.sqrt(((V-A_neg)**2).sum(axis=1))
    C = S_neg/(S_star+S_neg)
    out = regions[["region_name_vi","digital_index_0_100","ai_readiness_0_100"]].copy()
    out["TOPSIS_score"] = C
    out["Rank"] = out["TOPSIS_score"].rank(ascending=False, method="dense").astype(int)
    return out.sort_values("TOPSIS_score", ascending=False), w


def page_bai6():
    section_title("Cấp độ trung bình", "Bài 6 — TOPSIS xếp hạng 6 vùng", "So sánh trọng số chuyên gia và trọng số khách quan Entropy cho ưu tiên đầu tư AI.")
    mode = st.radio("Chế độ trọng số", ["Chuyên gia", "Entropy"], horizontal=True)
    df, w = topsis_scores(entropy=(mode=="Entropy"))
    k1,k2,k3 = st.columns(3)
    for col, (_, row) in zip([k1,k2,k3], df.head(3).iterrows()):
        with col: metric_card(f"Top {row['Rank']}", row["region_name_vi"], f"Score {row['TOPSIS_score']:.3f}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.plotly_chart(plot_bar(df.sort_values("TOPSIS_score"), "TOPSIS_score", "region_name_vi", f"TOPSIS — {mode}", orientation="h", text="TOPSIS_score"), use_container_width=True)
    wdf = pd.DataFrame({"Tiêu chí":["GRDP/ng", "FDI", "Digital", "AI", "LĐĐT", "R&D", "Internet", "Gini"], "Trọng số": w})
    st.plotly_chart(plot_bar(wdf, "Tiêu chí", "Trọng số", "Bộ trọng số đang dùng", text="Trọng số"), use_container_width=True)
    agent_box("Bài 6", [
        f"Theo chế độ {mode}, vùng dẫn đầu là {df.iloc[0]['region_name_vi']}.",
        "Entropy thường làm trọng số tăng ở tiêu chí có độ phân tán dữ liệu lớn, vì tiêu chí đó mang nhiều thông tin phân biệt hơn.",
        "Khi chọn 3 trung tâm AI, nên lấy TOPSIS làm lõi định lượng rồi bổ sung tiêu chí địa-chính trị và an ninh dữ liệu."
    ])


# -----------------------------------------------------------------------------
# BÀI 7
# -----------------------------------------------------------------------------
def pareto_mask(F):
    # Minimization mask
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


@st.cache_data(show_spinner=False)
def generate_pareto(seed=42, samples=1500):
    rng = np.random.default_rng(seed)
    rows, F = [], []
    e = np.array([0.42,0.55,0.48,0.32,0.62,0.38])
    rho = np.array([0.18,0.45,0.28,0.12,0.52,0.22])
    sig = np.array([0.32,0.28,0.30,0.35,0.25,0.30])
    for _ in range(samples):
        base = np.ones(6)*5000
        rem = 50000 - base.sum()
        add = rng.dirichlet(np.ones(6))*rem
        sums = np.minimum(base+add, 12000)
        deficit = 50000 - sums.sum()
        if deficit > 1e-6:
            room = 12000 - sums
            if room.sum() > 0:
                sums += room/room.sum()*deficit
        X = np.vstack([rng.dirichlet(np.ones(4))*s for s in sums])
        if X[:,3].sum() < 12000:
            need = 12000 - X[:,3].sum()
            donors = X[:,:3].sum()
            if donors > need:
                for r in range(6):
                    take = need * X[r,:3].sum()/donors
                    shares = X[r,:3]/max(X[r,:3].sum(), 1e-9)
                    X[r,:3] -= take*shares
                    X[r,3] += take
        gdp = (BETA*X).sum()
        region_sums = X.sum(axis=1)
        inequality = np.abs(region_sums-region_sums.mean()).mean()
        emission = (e*(X[:,0]+X[:,2])).sum()
        risk = (rho*X[:,2]).sum() - (sig*X[:,3]).sum()
        rows.append(X)
        F.append([-gdp, inequality, emission, risk])
    F = np.array(F)
    mask = pareto_mask(F)
    pareto = F[mask]
    Xs = np.array(rows, dtype=object)[mask]
    out = pd.DataFrame({
        "GDP_gain": -pareto[:,0], "Inequality": pareto[:,1], "Emission": pareto[:,2], "SecurityRisk": pareto[:,3]
    })
    return out, list(Xs)


def page_bai7():
    section_title("Cấp độ khá khó", "Bài 7 — Pareto đa mục tiêu", "Mô phỏng tập nghiệm Pareto cho 4 mục tiêu: tăng trưởng, bao trùm, môi trường, an ninh dữ liệu.")
    samples = st.slider("Số nghiệm mô phỏng", 500, 3000, 1500, 250)
    out, Xs = generate_pareto(samples=samples)
    weights = np.array([0.40,0.25,0.20,0.15])
    norm = out.copy()
    norm["GDP_gain"] = minmax_good(norm["GDP_gain"])
    for col in ["Inequality","Emission","SecurityRisk"]:
        norm[col] = minmax_bad(norm[col])
    score = norm.to_numpy() @ weights
    best_idx = int(np.argmax(score))
    k1,k2,k3,k4 = st.columns(4)
    with k1: metric_card("Số nghiệm Pareto", str(len(out)))
    with k2: metric_card("GDP gain compromise", f"{out.iloc[best_idx]['GDP_gain']:,.0f}")
    with k3: metric_card("Emission", f"{out.iloc[best_idx]['Emission']:,.0f}")
    with k4: metric_card("Security risk", f"{out.iloc[best_idx]['SecurityRisk']:,.0f}")
    fig = px.scatter_3d(out, x="GDP_gain", y="Inequality", z="Emission", color="SecurityRisk", template="plotly_dark", title="Biên Pareto mô phỏng 3D")
    fig.update_layout(height=620, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(out.iloc[[best_idx]].assign(TOPSIS_compromise_score=score[best_idx]), use_container_width=True, hide_index=True)
    agent_box("Bài 7", [
        "Không có phương án thắng tuyệt đối: tăng GDP thường kéo theo phát thải hoặc rủi ro dữ liệu cao hơn.",
        "Nghiệm compromise được chọn bằng TOPSIS theo trọng số chính sách, vì vậy thay đổi trọng số sẽ đổi khuyến nghị.",
        "Đây là phần giúp bài làm nhìn có chiều sâu: mô hình không thay quyết định chính trị, mà phơi bày đánh đổi để người ra quyết định chọn."
    ])


# -----------------------------------------------------------------------------
# BÀI 8
# -----------------------------------------------------------------------------
def simulate_strategy(alloc, shock=False):
    years = np.arange(2026, 2036)
    K, D, AI, H, A, L = 27500.0, 20.3, 86.0, 30.0, cobb_data()[3], 53.9
    rows = []
    for t, y in enumerate(years):
        budget = 900
        a = np.array(alloc(t))
        Y = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
        if shock and y == 2028:
            Y *= 0.92
            budget *= 0.80
        C = max(Y - budget, 1)
        rows.append([y,K,D,AI,H,A,Y,C,*a])
        K = (1-0.05)*K + a[0]*budget
        D = (1-0.12)*D + a[1]*budget/18
        AI = (1-0.15)*AI + a[2]*budget/9
        H = H + 0.8*a[3]*budget/120 - 0.02*H
        A = A*(1+0.003*D/100+0.002*AI/100+0.004*H/100)
        L *= 1.005
    return pd.DataFrame(rows, columns=["Năm","K","D","AI","H","A","Y","C","aK","aD","aAI","aH"])


def page_bai8():
    section_title("Cấp độ khá khó", "Bài 8 — Tối ưu động 2026-2035", "Mô phỏng quỹ đạo liên thời gian của K, D, AI, H, Y và C dưới nhiều chiến lược phân bổ.")
    strategies = {
        "Cân bằng": lambda t: [0.40,0.25,0.15,0.20],
        "Front-load số hóa": lambda t: [0.25,0.45 if t<3 else 0.25,0.15,0.15 if t<3 else 0.35],
        "AI dẫn dắt": lambda t: [0.20,0.20,0.45,0.15],
        "Nhân lực đi trước": lambda t: [0.30,0.20,0.10 if t<4 else 0.25,0.40 if t<4 else 0.25],
    }
    name = st.selectbox("Chọn chiến lược", list(strategies.keys()))
    shock = st.toggle("Thêm cú sốc năm 2028: Y giảm 8%", value=False)
    df = simulate_strategy(strategies[name], shock)
    k1,k2,k3 = st.columns(3)
    with k1: metric_card("GDP 2035", f"{df.Y.iloc[-1]:,.0f}")
    with k2: metric_card("H 2035", f"{df.H.iloc[-1]:.1f}%")
    with k3: metric_card("AI 2035", f"{df.AI.iloc[-1]:.1f}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    for col in ["Y","K","D","AI","H","C"]:
        st.plotly_chart(plot_line(df, "Năm", col, f"Quỹ đạo {col} — {name}"), use_container_width=True)
    comp = []
    for n, f in strategies.items():
        tmp = simulate_strategy(f, shock)
        welfare = sum((0.97**i)*math.log(max(c,1)) for i,c in enumerate(tmp.C))
        comp.append([n, tmp.Y.iloc[-1], tmp.H.iloc[-1], welfare])
    cdf = pd.DataFrame(comp, columns=["Chiến lược","GDP2035","H2035","Welfare_log"])
    st.dataframe(cdf.sort_values("Welfare_log", ascending=False), use_container_width=True, hide_index=True)
    agent_box("Bài 8", [
        f"Chiến lược đang xem là {name}; cú sốc 2028 {'đang bật' if shock else 'đang tắt'}.",
        "Front-load giúp tăng năng lực sớm nhưng có thể làm tiêu dùng C thấp hơn ở giai đoạn đầu.",
        "Chiến lược nhân lực đi trước thường bền hơn khi có cú sốc vì H làm nền cho hấp thụ công nghệ và giảm rủi ro điều chỉnh."
    ])


# -----------------------------------------------------------------------------
# BÀI 9
# -----------------------------------------------------------------------------
def solve_labor(cap=6000, budget=30000):
    names = ["Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng","Bán buôn-bán lẻ","Tài chính-Ngân hàng","Logistics-Vận tải","CNTT-Truyền thông","Giáo dục-Đào tạo"]
    risk = np.array([18,42,25,38,52,35,28,22])/100
    a1 = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
    b1 = np.array([45,28,35,32,22,30,20,55])
    c1 = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
    d1 = np.array([50,32,42,38,26,36,24,62])
    coef_ai = a1 - c1*risk
    coef_h = b1
    c = -np.r_[coef_ai, coef_h]
    A, b = [], []
    A.append(np.ones(16)); b.append(budget)
    for i in range(8):
        row = np.zeros(16); row[i] = -coef_ai[i]; row[8+i] = -coef_h[i]; A.append(row); b.append(0)
        row = np.zeros(16); row[i] = c1[i]*risk[i]; row[8+i] = -d1[i]; A.append(row); b.append(0)
    bounds = [(0, cap)]*16
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    if not res.success:
        return None
    xAI, xH = res.x[:8], res.x[8:]
    df = pd.DataFrame({"Ngành":names,"x_AI":xAI,"x_H":xH})
    df["NewJob"] = a1*xAI
    df["UpgradeJob"] = b1*xH
    df["DisplacedJob"] = c1*risk*xAI
    df["RetrainingCapacity"] = d1*xH
    df["NetJob"] = df.NewJob + df.UpgradeJob - df.DisplacedJob
    return df


def page_bai9():
    section_title("Cấp độ khá khó", "Bài 9 — Tác động AI tới lao động", "Tối ưu phân bổ AI và đào tạo lại để NetJob ròng không âm theo ngành.")
    c1,c2 = st.columns(2)
    with c1: budget = st.slider("Ngân sách lao động-AI", 15000, 50000, 30000, 1000)
    with c2: cap = st.slider("Trần hấp thụ mỗi biến/ngành", 2000, 10000, 6000, 500)
    df = solve_labor(cap, budget)
    if df is None:
        st.error("Bài toán không khả thi.")
        return
    k1,k2,k3 = st.columns(3)
    with k1: metric_card("Tổng NetJob", f"{df.NetJob.sum():,.0f}")
    with k2: metric_card("Đầu tư AI", f"{df.x_AI.sum():,.0f}")
    with k3: metric_card("Đào tạo lại H", f"{df.x_H.sum():,.0f}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    long = df.melt(id_vars="Ngành", value_vars=["x_AI","x_H"], var_name="Hạng mục", value_name="Đầu tư")
    fig = px.bar(long, x="Ngành", y="Đầu tư", color="Hạng mục", barmode="group", template="plotly_dark", title="Phân bổ AI và đào tạo lại theo ngành")
    fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(plot_bar(df.sort_values("NetJob"), "NetJob", "Ngành", "NetJob ròng theo ngành", orientation="h", text="NetJob"), use_container_width=True)
    agent_box("Bài 9", [
        "Ngành có rủi ro tự động hóa cao cần đi kèm x_H để DisplacedJob không vượt năng lực đào tạo lại.",
        "Nếu app phân bổ nhiều vào H, đó là vì mô hình đang ưu tiên NetJob ròng và tránh tổn thương lao động.",
        "Thông điệp chính sách: tốc độ tự động hóa không nên vượt tốc độ đào tạo lại."
    ])


# -----------------------------------------------------------------------------
# BÀI 10
# -----------------------------------------------------------------------------
def solve_sp(h_floor_ratio=0.0):
    beta = np.array([1.00,1.10,1.25,0.95])
    beta_s = np.array([[1.25,1.35,1.55,1.05],[1.00,1.10,1.25,0.95],[0.75,0.85,0.90,1.00],[0.40,0.50,0.55,1.10]])
    p = np.array([0.30,0.45,0.20,0.05])
    n = 4 + 16
    c = np.zeros(n); c[:4] = -beta
    for s in range(4): c[4+s*4:4+(s+1)*4] = -p[s]*beta_s[s]
    A,b = [], []
    row = np.zeros(n); row[:4]=1; A.append(row); b.append(65000)
    if h_floor_ratio>0:
        row = np.zeros(n); row[3] = -1; A.append(row); b.append(-65000*h_floor_ratio)
    for s in range(4):
        row = np.zeros(n); row[4+s*4:4+(s+1)*4]=1; A.append(row); b.append(15000)
        row = np.zeros(n); row[4+s*4+2]=1; row[3] = -0.5; A.append(row); b.append(0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0,None)]*n, method="highs")
    return res, beta, beta_s, p


def page_bai10():
    section_title("Cấp độ khó", "Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn", "First-stage here-and-now, second-stage recourse theo 4 kịch bản kinh tế toàn cầu.")
    hratio = st.slider("Ràng buộc quản trị: H tối thiểu ở giai đoạn 1", 0.00, 0.30, 0.10, 0.01)
    res,beta,beta_s,p = solve_sp(hratio)
    if not res.success:
        st.error("Không giải được mô hình SP.")
        return
    x = res.x[:4]; y = res.x[4:].reshape(4,4)
    items = ["I","D","AI","H"]
    scenarios = ["Lạc quan","Cơ sở","Bi quan","Khủng hoảng"]
    df1 = pd.DataFrame({"Hạng mục":items,"First-stage x":x})
    df2 = pd.DataFrame(y, columns=items, index=scenarios).reset_index(names="Kịch bản")
    k1,k2,k3 = st.columns(3)
    with k1: metric_card("Objective SP", f"{-res.fun:,.0f}")
    with k2: metric_card("First-stage total", f"{x.sum():,.0f}")
    with k3: metric_card("H stage 1", f"{x[3]:,.0f}")
    st.dataframe(df1, use_container_width=True, hide_index=True)
    st.dataframe(df2, use_container_width=True, hide_index=True)
    st.plotly_chart(plot_bar(df1, "Hạng mục", "First-stage x", "Quyết định first-stage"), use_container_width=True)
    # Perfect information approximation: solve each scenario independently x+y <= 80000, y removed equivalent all budget into best scenario coef.
    pi_vals = []
    for s in range(4):
        pi_vals.append(80000*max(beta_s[s]))
    evpi = np.dot(p,pi_vals) - (-res.fun)
    st.info(f"EVPI xấp xỉ: **{evpi:,.0f}**. Giá trị càng cao nghĩa là thông tin hoàn hảo về kịch bản tương lai càng đáng giá.")
    agent_box("Bài 10", [
        "SP giúp giữ một phần linh hoạt cho giai đoạn 2 thay vì khóa toàn bộ ngân sách ngay từ đầu.",
        "Trong kịch bản khủng hoảng, hệ số H cao hơn nên vốn nhân lực đóng vai trò như bảo hiểm chính sách.",
        "Nếu thêm sàn H, nghiệm thực tế hơn vì hạn chế mô hình dồn quá nhiều vào hạng mục có hệ số ngắn hạn cao nhất."
    ])


# -----------------------------------------------------------------------------
# BÀI 11
# -----------------------------------------------------------------------------
ACTIONS = {
    0:("Truyền thống", np.array([0.70,0.10,0.10,0.10])),
    1:("Cân bằng", np.array([0.40,0.25,0.15,0.20])),
    2:("Số hóa nhanh", np.array([0.25,0.45,0.15,0.15])),
    3:("AI dẫn dắt", np.array([0.20,0.20,0.45,0.15])),
    4:("Bao trùm", np.array([0.30,0.20,0.10,0.40])),
}


def discretize(growth, digital, ai, unemp):
    return np.array([
        0 if growth<5 else 1 if growth<8 else 2,
        0 if digital<18 else 1 if digital<28 else 2,
        0 if ai<80 else 1 if ai<120 else 2,
        0 if unemp<3 else 1 if unemp<5 else 2,
    ], dtype=int)


@st.cache_data(show_spinner=False)
def train_q(seed=7, episodes=2500):
    rng = np.random.default_rng(seed)
    Q = np.zeros((3,3,3,3,5))
    rewards = []
    for ep in range(episodes):
        growth, digital, ai, unemp = 6.5, 20.3, 86.0, 4.0
        s = discretize(growth,digital,ai,unemp)
        total = 0
        eps = max(0.05, 1-ep/(episodes*0.65))
        for t in range(10):
            if rng.random() < eps:
                a = rng.integers(0,5)
            else:
                a = int(np.argmax(Q[tuple(s)]))
            alloc = ACTIONS[a][1]
            growth_delta = 1.4*alloc[0] + 2.2*alloc[1] + 2.6*alloc[2] + 1.2*alloc[3] + rng.normal(0,0.25)
            cyber = 3.5*alloc[2] - 1.2*alloc[3]
            emission = 2.0*alloc[0] + 2.5*alloc[2] - 0.5*alloc[1]
            unem_delta = 1.8*alloc[2] - 2.2*alloc[3] - 0.5*alloc[1] + rng.normal(0,0.12)
            reward = 0.40*growth_delta - 0.25*max(unem_delta,0) - 0.20*max(cyber,0) - 0.15*max(emission,0)
            total += reward
            growth = np.clip(growth + growth_delta - 1.0, 1, 12)
            digital = np.clip(digital + 4*alloc[1] + 1.2*alloc[2], 10, 45)
            ai = np.clip(ai + 18*alloc[2] + 5*alloc[3], 40, 180)
            unemp = np.clip(unemp + unem_delta, 1.5, 9)
            s2 = discretize(growth,digital,ai,unemp)
            old = Q[tuple(s)+(a,)]
            Q[tuple(s)+(a,)] = old + 0.10*(reward + 0.95*np.max(Q[tuple(s2)]) - old)
            s = s2
        rewards.append(total)
    return Q, pd.DataFrame({"Episode":np.arange(episodes), "Reward":rewards})


def page_bai11():
    section_title("Cấp độ khó", "Bài 11 — Q-learning chính sách thích nghi", "Huấn luyện Q-table 81 trạng thái × 5 hành động để chọn chính sách theo trạng thái kinh tế.")
    episodes = st.slider("Số episodes huấn luyện", 500, 5000, 2500, 500)
    Q, rewards = train_q(episodes=episodes)
    st.plotly_chart(plot_line(rewards.rolling(50, min_periods=1).mean(), "Episode", "Reward", "Learning curve — reward trung bình trượt 50 episodes"), use_container_width=True)
    examples = pd.DataFrame([
        ["Việt Nam 2026", 6.5,20.3,86,4.0],
        ["GDP thấp - D thấp - U cao", 3.5,14,60,7.0],
        ["GDP cao - AI cao - U thấp", 9.0,32,140,2.5],
        ["Số hóa tốt nhưng AI yếu", 7.0,30,65,4.0],
        ["Khủng hoảng lao động", 4.5,22,115,8.0],
    ], columns=["Trạng thái","Growth","Digital","AI","Unemployment"])
    recs = []
    for _, row in examples.iterrows():
        s = discretize(row.Growth,row.Digital,row.AI,row.Unemployment)
        a = int(np.argmax(Q[tuple(s)]))
        recs.append([row["Trạng thái"], tuple(s), ACTIONS[a][0], a])
    rdf = pd.DataFrame(recs, columns=["Trạng thái","Mã trạng thái","Hành động π*","Action id"])
    st.dataframe(rdf, use_container_width=True, hide_index=True)
    agent_box("Bài 11", [
        "Q-learning chọn hành động theo trạng thái, nên linh hoạt hơn chính sách cố định kiểu luôn chọn một phương án.",
        "Khi thất nghiệp cao, chính sách bao trùm/nhân lực thường được ưu tiên vì reward phạt rủi ro lao động.",
        "Phần này chỉ minh họa kỹ thuật; quyết định chính sách thật vẫn cần thẩm định, tranh luận và trách nhiệm con người."
    ])


# -----------------------------------------------------------------------------
# BÀI 12
# -----------------------------------------------------------------------------
def scenario_2030(name, alloc):
    df = simulate_strategy(lambda t: alloc, False)
    final = df[df["Năm"]==2030].iloc[0]
    risk = alloc[2]*55 - alloc[3]*38 + alloc[0]*12
    inclusiveness = alloc[3]*60 + alloc[1]*20 - alloc[2]*10
    return {"Kịch bản":name,"GDP_2030":final.Y,"Digital_2030":final.D,"AI_2030":final.AI,"H_2030":final.H,"RiskIndex":risk,"InclusionIndex":inclusiveness}


def page_bai12():
    section_title("Đồ án tích hợp", "Bài 12 — Dashboard AIDEOM-VN", "Tích hợp 6 module: macro, sẵn sàng số, tối ưu phân bổ, lao động, rủi ro và dashboard ra quyết định.")
    scenarios = {
        "S1 Truyền thống": np.array([0.70,0.10,0.10,0.10]),
        "S2 Số hóa nhanh": np.array([0.25,0.45,0.15,0.15]),
        "S3 AI dẫn dắt": np.array([0.20,0.20,0.45,0.15]),
        "S4 Bao trùm số": np.array([0.30,0.20,0.10,0.40]),
        "S5 Tối ưu cân bằng": np.array([0.35,0.25,0.20,0.20]),
    }
    summary = pd.DataFrame([scenario_2030(k,v) for k,v in scenarios.items()])
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Tổng quan", "Phân bổ", "Kịch bản so sánh", "Cảnh báo rủi ro", "Khuyến nghị"])
    with tab1:
        k1,k2,k3,k4 = st.columns(4)
        best = summary.loc[summary.GDP_2030.idxmax()]
        safe = summary.loc[summary.RiskIndex.idxmin()]
        inc = summary.loc[summary.InclusionIndex.idxmax()]
        with k1: metric_card("GDP 2030 cao nhất", best["Kịch bản"], f"{best.GDP_2030:,.0f}")
        with k2: metric_card("Rủi ro thấp nhất", safe["Kịch bản"], f"{safe.RiskIndex:.1f}")
        with k3: metric_card("Bao trùm cao nhất", inc["Kịch bản"], f"{inc.InclusionIndex:.1f}")
        with k4: metric_card("Số kịch bản", "5")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with tab2:
        alloc_df = pd.DataFrame(scenarios, index=["K","D","AI","H"]).T.reset_index(names="Kịch bản")
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
        long = alloc_df.melt(id_vars="Kịch bản", var_name="Hạng mục", value_name="Tỷ trọng")
        fig = px.bar(long, x="Kịch bản", y="Tỷ trọng", color="Hạng mục", barmode="stack", template="plotly_dark", title="Cơ cấu phân bổ 5 kịch bản")
        fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.plotly_chart(plot_bar(summary, "Kịch bản", "GDP_2030", "So sánh GDP 2030", text="GDP_2030"), use_container_width=True)
        radar = go.Figure()
        cats = ["GDP_2030","Digital_2030","AI_2030","H_2030","InclusionIndex"]
        norm = summary.copy()
        for c in cats: norm[c] = minmax_good(norm[c])
        for _, row in norm.iterrows():
            radar.add_trace(go.Scatterpolar(r=[row[c] for c in cats], theta=cats, fill="toself", name=row["Kịch bản"]))
        radar.update_layout(template="plotly_dark", title="Radar chuẩn hóa KPI", height=560, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(radar, use_container_width=True)
    with tab4:
        alert = summary.copy()
        alert["Mức cảnh báo"] = pd.cut(alert.RiskIndex, bins=[-99,10,25,99], labels=["Thấp","Trung bình","Cao"])
        st.dataframe(alert[["Kịch bản","RiskIndex","Mức cảnh báo","InclusionIndex"]], use_container_width=True, hide_index=True)
        st.plotly_chart(plot_bar(alert, "Kịch bản", "RiskIndex", "Chỉ số rủi ro theo kịch bản", text="RiskIndex"), use_container_width=True)
    with tab5:
        st.markdown("""
<div class="card">
<h3>Khuyến nghị chính sách ngắn gọn</h3>
<p class="subtle">S5 Tối ưu cân bằng là phương án dễ bảo vệ nhất trong bài thuyết trình vì không cực đoan như S3 AI dẫn dắt, nhưng vẫn tốt hơn S1 truyền thống về số hóa và năng lực AI. Nếu hội đồng hỏi về tính bao trùm, dùng S4 làm phương án xã hội; nếu hỏi về tăng trưởng nhanh, dùng S3 làm phương án tham chiếu rủi ro cao.</p>
</div>
""", unsafe_allow_html=True)
    agent_box("Bài 12", [
        "Dashboard tích hợp biến các bài rời rạc thành 5 kịch bản có thể so sánh trên cùng hệ KPI.",
        "S5 là lựa chọn trình bày an toàn: cân bằng tăng trưởng, nhân lực, số hóa và rủi ro.",
        "Khi bảo vệ, nên nhấn mạnh app là công cụ hỗ trợ ra quyết định, không phải máy tự quyết định chính sách."
    ])


# -----------------------------------------------------------------------------
# HOME & NAVIGATION
# -----------------------------------------------------------------------------
def page_home():
    section_title("VN AIDEOM-VN", "AI-Driven Decision Optimization Model for Vietnam", "Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI — dùng dữ liệu thực 2020-2025.")
    latest = macro.sort_values("year").iloc[-1]
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("GDP 2025", f"{latest.GDP_billion_USD:,.1f} tỷ USD", f"↑ {latest.GDP_growth_pct:.2f}%")
    with c2: metric_card("Kinh tế số/GDP", f"≈{latest.digital_economy_share_GDP_pct:.1f}%", "mục tiêu 2030: 30%")
    with c3: metric_card("FDI giải ngân", f"{latest.FDI_disbursed_billion_USD:.1f} tỷ USD")
    with c4: metric_card("GDP/người", f"{latest.GDP_per_capita_USD:,.0f} USD")

    st.markdown("## 📚 12 bài toán theo 4 cấp độ")
    levels = [
        ("🟢 Cấp độ DỄ — Làm quen mô hình", [("Bài 1", "Hàm sản xuất Cobb-Douglas mở rộng + AI"), ("Bài 2", "LP phân bổ ngân sách 4 hạng mục"), ("Bài 3", "Priority 10 ngành")]),
        ("🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển", [("Bài 4", "LP ngành-vùng"), ("Bài 5", "MIP chọn 15 dự án"), ("Bài 6", "TOPSIS 6 vùng")]),
        ("🟠 Cấp độ KHÁ KHÓ — Đánh đổi & động học", [("Bài 7", "Pareto đa mục tiêu"), ("Bài 8", "Động 2026-2035"), ("Bài 9", "Lao động & AI")]),
        ("🔴 Cấp độ KHÓ — Bất định, RL, tích hợp", [("Bài 10", "Stochastic SP"), ("Bài 11", "Q-learning RL"), ("Bài 12", "AIDEOM tích hợp")]),
    ]
    for title, rows in levels:
        with st.expander(title, expanded=True):
            st.dataframe(pd.DataFrame(rows, columns=["Bài", "Nội dung"]), use_container_width=True, hide_index=True)

    st.markdown("## 🧭 Luồng xử lý dữ liệu")
    st.markdown("""
<div class="card">
<span class="badge">Data</span> Macro 2020-2025 · Sectors 2024 · Regions 2024<br><br>
<span class="badge">Model</span> Cobb-Douglas · LP · MIP · TOPSIS · Pareto · Dynamic · Labor · Stochastic · Q-learning<br><br>
<span class="badge">Output</span> Bảng kết quả · Biểu đồ · Kịch bản · Tác nhân phân tích chính sách
</div>
""", unsafe_allow_html=True)

PAGES = {
    "🏠 Trang chủ": page_home,
    "🌱 Bài 1 — Cobb-Douglas + AI": page_bai1,
    "💰 Bài 2 — LP ngân sách số": page_bai2,
    "📊 Bài 3 — Priority 10 ngành": page_bai3,
    "🗺️ Bài 4 — LP ngành-vùng": page_bai4,
    "🎯 Bài 5 — MIP 15 dự án": page_bai5,
    "🏆 Bài 6 — TOPSIS 6 vùng": page_bai6,
    "🌐 Bài 7 — Pareto đa mục tiêu": page_bai7,
    "⏳ Bài 8 — Động 2026-2035": page_bai8,
    "👷 Bài 9 — Lao động & AI": page_bai9,
    "🎲 Bài 10 — Stochastic SP": page_bai10,
    "🧠 Bài 11 — Q-learning RL": page_bai11,
    "VN Bài 12 — AIDEOM tích hợp": page_bai12,
}

with st.sidebar:
    st.markdown("### VN AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    selected = st.radio("Menu", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dữ liệu:** NSO/GSO · MoST · MIC · MPI · WB · GII")
    st.caption("Bản nộp mẫu — có thể đổi tên folder theo MSV_HọTên trước khi nén.")

PAGES[selected]()
