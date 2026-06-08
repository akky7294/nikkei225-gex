"""
Nikkei 225 Options Gamma Exposure (GEX) Visualizer
JPX公式PDFデータを使用してガンマエクスポージャーを可視化する
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
import io
import re
import requests
from datetime import date
from pathlib import Path
import zipfile


@st.cache_data(ttl=60)  # 1分キャッシュ
def fetch_nikkei_spot() -> float:
    """Yahoo FinanceからリアルタイムのNikkei225現値を取得する"""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EN225?interval=1m&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return float(price)
    except Exception:
        return None


@st.cache_data(ttl=3600)  # 1時間キャッシュ
def fetch_japan_rate() -> float:
    """日本の短期金利（10年国債利回り）をYahoo Financeから取得する"""
    try:
        # 日本10年国債利回り
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?interval=1d&range=5d"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        rate = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return float(rate) / 100  # % → 小数
    except Exception:
        return None

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.set_page_config(
    page_title="日経225 Gamma Exposure",
    page_icon="📊",
    layout="wide",
)

# ─── Black-Scholes 関数群 ────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def implied_vol(S, K, T, r, market_price, option_type="call", default=0.20):
    if T <= 0 or market_price <= 0:
        return default
    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if market_price <= intrinsic * 1.001:
        return default
    try:
        iv = brentq(
            lambda s: bs_price(S, K, T, r, s, option_type) - market_price,
            1e-6, 5.0, xtol=1e-5, maxiter=100
        )
        return iv if 0.01 <= iv <= 5.0 else default
    except Exception:
        return default


# ─── GEX計算 ─────────────────────────────────────────────────────────────────

def calculate_gex(df: pd.DataFrame, spot: float, r: float = 0.001) -> pd.DataFrame:
    MULTIPLIER = 1000
    records = []
    for _, row in df.iterrows():
        T = row["days_to_expiry"] / 365.0
        gamma = bs_gamma(spot, row["strike"], T, r, row["iv"])
        gex = gamma * row["oi"] * MULTIPLIER * spot**2
        sign = 1 if row["type"] == "call" else -1
        records.append({
            "strike": row["strike"],
            "expiry": row["expiry"],
            "type": row["type"],
            "gex": sign * gex,
            "call_gex": gex if row["type"] == "call" else 0,
            "put_gex": -gex if row["type"] == "put" else 0,
            "gamma": gamma,
            "oi": row["oi"],
            "iv": row["iv"],
        })
    return pd.DataFrame(records)


# ─── JPX PDF パーサー ─────────────────────────────────────────────────────────

def parse_jpx_pdf(raw: bytes, today: date, spot: float, r: float) -> pd.DataFrame:
    if not PDF_AVAILABLE:
        st.error("pdfplumberが必要です。")
        return pd.DataFrame()

    row_pattern = re.compile(r'(20\d{4})\s+(\d{2}\.\d{2})\s+([\d,]+)\s+\d{6,12}(.*)')
    records = []
    current_type = None

    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "PutOptions" in text:
                current_type = "put"
            if "CallOptions" in text:
                current_type = "call"
            if current_type is None:
                continue

            for line in text.split("\n"):
                m = row_pattern.match(line.strip())
                if not m:
                    continue
                contract_ym = m.group(1)
                exp_md = m.group(2)
                strike_str = m.group(3).replace(",", "")
                rest = m.group(4).strip().split()

                try:
                    strike = int(strike_str)
                except ValueError:
                    continue

                oi = 0
                for token in reversed(rest):
                    clean = token.replace(",", "")
                    if re.fullmatch(r'\d+', clean):
                        oi = int(clean)
                        break
                if oi == 0:
                    continue

                settlement = 0.0
                if len(rest) >= 3:
                    try:
                        v = float(rest[-3].replace(",", ""))
                        if v > 0:
                            settlement = v
                    except ValueError:
                        pass

                year = int(contract_ym[:4])
                try:
                    exp_month = int(exp_md.split(".")[0])
                    exp_day = int(exp_md.split(".")[1])
                    expiry = date(year, exp_month, exp_day)
                except (ValueError, IndexError):
                    continue

                days = (expiry - today).days
                if days <= 0:
                    continue

                T = days / 365.0
                iv = implied_vol(spot, strike, T, r, settlement, current_type) if settlement > 0 else 0.20

                records.append({
                    "strike": strike,
                    "expiry": pd.Timestamp(expiry),
                    "type": current_type,
                    "oi": oi,
                    "iv": iv,
                    "days_to_expiry": days,
                    "settlement": settlement,
                })

    if not records:
        st.error("PDFからデータを抽出できませんでした。")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    n_iv = (df["iv"] != 0.20).sum()
    st.success(
        f"PDF読み込み完了: {len(df)}行 "
        f"（Call:{len(df[df.type=='call'])} / Put:{len(df[df.type=='put'])}）"
        f"  IV実測: {n_iv}行 / 推定: {len(df)-n_iv}行"
    )
    return df


# ─── デモデータ生成 ───────────────────────────────────────────────────────────

def generate_demo_data(spot: float, today: date) -> pd.DataFrame:
    np.random.seed(42)
    strikes = np.arange(
        round(spot * 0.85 / 500) * 500,
        round(spot * 1.15 / 500) * 500 + 500,
        500,
    )
    expiries_days = [14, 42, 77]
    rows = []
    for days in expiries_days:
        expiry = pd.Timestamp(today) + pd.Timedelta(days=days)
        atm_iv = 0.20
        for strike in strikes:
            moneyness = strike / spot
            skew = 0.05 * abs(moneyness - 1.0) + 0.02 * max(0, 1.0 - moneyness)
            iv = atm_iv + skew
            call_oi = int(np.exp(-3 * max(0, moneyness - 1.0)**2) * 5000 * np.random.uniform(0.7, 1.3))
            put_oi = int(np.exp(-3 * max(0, 1.0 - moneyness)**2) * 5000 * np.random.uniform(0.7, 1.3))
            rows.append({"strike": strike, "expiry": expiry, "type": "call", "oi": call_oi, "iv": iv, "days_to_expiry": days})
            rows.append({"strike": strike, "expiry": expiry, "type": "put", "oi": put_oi, "iv": iv, "days_to_expiry": days})
    return pd.DataFrame(rows)


# ─── メインチャート描画 ───────────────────────────────────────────────────────

def build_gex_chart(gex_df: pd.DataFrame, spot: float, selected_expiry, oi_threshold: int):
    """
    Tiger Brokers風GEXチャート:
    - コール(赤バー) / プット(緑バー) を別々に表示
    - アグリゲートGEX累積ライン(青・右軸)
    - プットウォール / コールウォール ラベル
    - 正ゾーン(緑背景) / 負ゾーン(赤背景)
    - ガンマフリップライン
    """
    # データ集計
    if selected_expiry == "全満期合算":
        agg = gex_df.groupby("strike", as_index=False).agg(
            call_gex=("call_gex", "sum"),
            put_gex=("put_gex", "sum"),
            gex=("gex", "sum"),
            oi=("oi", "sum"),
        )
    else:
        filtered = gex_df[gex_df["expiry"] == selected_expiry]
        agg = filtered.groupby("strike", as_index=False).agg(
            call_gex=("call_gex", "sum"),
            put_gex=("put_gex", "sum"),
            gex=("gex", "sum"),
            oi=("oi", "sum"),
        )

    # 現値±25%・OIフィルター
    agg = agg[
        (agg["strike"] >= spot * 0.75) &
        (agg["strike"] <= spot * 1.25) &
        (agg["oi"] >= oi_threshold)
    ].sort_values("strike").reset_index(drop=True)

    if agg.empty:
        return None, 0, None, None, None

    # アグリゲートGEX（累積）
    agg["agg_gex"] = agg["gex"].cumsum()

    # ガンマフリップ（累積GEXがゼロ交差する点）
    gamma_flip = None
    for i in range(1, len(agg)):
        if agg["agg_gex"].iloc[i - 1] * agg["agg_gex"].iloc[i] < 0:
            gamma_flip = int(agg["strike"].iloc[i])
            break

    # プットウォール（最大プットGEX絶対値）
    put_wall_idx = agg["put_gex"].abs().idxmax()
    put_wall = int(agg.loc[put_wall_idx, "strike"])

    # コールウォール（最大コールGEX）
    call_wall_idx = agg["call_gex"].idxmax()
    call_wall = int(agg.loc[call_wall_idx, "strike"])

    net_total = agg["gex"].sum()
    unit = 1e8  # 億円

    # ─── Figure（2軸）───
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    strikes = agg["strike"].tolist()

    # ── 背景ゾーン（正=薄緑, 負=薄赤）──
    flip_idx = None
    if gamma_flip is not None:
        flip_idx = agg[agg["strike"] == gamma_flip].index[0] if gamma_flip in agg["strike"].values else None

    # 正ゾーン
    pos_zone = agg[agg["agg_gex"] >= 0]
    neg_zone = agg[agg["agg_gex"] < 0]

    for zone_df, fillcolor in [(pos_zone, "rgba(46,204,113,0.08)"), (neg_zone, "rgba(231,76,60,0.08)")]:
        if not zone_df.empty:
            x0 = zone_df["strike"].min() - 125
            x1 = zone_df["strike"].max() + 125
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=fillcolor,
                layer="below",
                line_width=0,
            )

    # ── プットGEX バー（緑）──
    fig.add_trace(
        go.Bar(
            x=agg["strike"],
            y=agg["put_gex"] / unit,
            name="プット GEX",
            marker_color="#27ae60",
            opacity=0.85,
            hovertemplate="Strike: %{x:,.0f}<br>Put GEX: %{y:.1f} 億円<extra></extra>",
        ),
        secondary_y=False,
    )

    # ── コールGEX バー（赤）──
    fig.add_trace(
        go.Bar(
            x=agg["strike"],
            y=agg["call_gex"] / unit,
            name="コール GEX",
            marker_color="#e74c3c",
            opacity=0.85,
            hovertemplate="Strike: %{x:,.0f}<br>Call GEX: %{y:.1f} 億円<extra></extra>",
        ),
        secondary_y=False,
    )

    # ── アグリゲートGEX ライン（青・右軸）──
    fig.add_trace(
        go.Scatter(
            x=agg["strike"],
            y=agg["agg_gex"] / unit,
            name="アグリゲートGEX",
            mode="lines",
            line=dict(color="#3498db", width=2),
            hovertemplate="Strike: %{x:,.0f}<br>Aggregate GEX: %{y:.1f} 億円<extra></extra>",
        ),
        secondary_y=True,
    )

    # ── 現値ライン ──
    fig.add_vline(
        x=spot,
        line_width=2, line_dash="solid", line_color="#f39c12",
        annotation_text=f"現値 {spot:,.0f}",
        annotation_font_color="#f39c12",
        annotation_position="top right",
    )

    # ── ガンマフリップライン ──
    if gamma_flip:
        fig.add_vline(
            x=gamma_flip,
            line_width=1.5, line_dash="dash", line_color="#f39c12",
            annotation_text=f"ガンマフリップ {gamma_flip:,.0f}",
            annotation_font_color="#f39c12",
            annotation_position="top left",
        )

    # ── プットウォール ──
    pw_y = float(agg.loc[put_wall_idx, "put_gex"]) / unit
    fig.add_annotation(
        x=put_wall, y=pw_y,
        text=f"▲ プットウォール {put_wall:,.0f}",
        showarrow=False,
        font=dict(color="#27ae60", size=11),
        yanchor="top" if pw_y < 0 else "bottom",
    )

    # ── コールウォール ──
    cw_y = float(agg.loc[call_wall_idx, "call_gex"]) / unit
    fig.add_annotation(
        x=call_wall, y=cw_y,
        text=f"▼ コールウォール {call_wall:,.0f}",
        showarrow=False,
        font=dict(color="#e74c3c", size=11),
        yanchor="bottom" if cw_y > 0 else "top",
    )

    # ── ゼロライン ──
    fig.add_hline(y=0, line_width=1, line_color="#555", secondary_y=False)
    fig.add_hline(y=0, line_width=0.5, line_color="#334", line_dash="dot", secondary_y=True)

    # x軸ティック
    x_min = int(agg["strike"].min())
    x_max = int(agg["strike"].max())
    tick_vals = list(range(round(x_min / 500) * 500, round(x_max / 500) * 500 + 500, 500))

    fig.update_layout(
        title=dict(
            text=f"日経225 Gamma Exposure  |  現値: {spot:,.0f}  |  Net GEX: {net_total/unit:.1f} 億円",
            font=dict(size=15),
        ),
        barmode="overlay",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=560,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=100),
        xaxis=dict(
            title="行使価格",
            tickvals=tick_vals,
            ticktext=[f"{v:,}" for v in tick_vals],
            tickangle=-45,
            gridcolor="#1a1a1a",
        ),
        yaxis=dict(
            title="GEX（億円）",
            gridcolor="#1a1a1a",
            zeroline=False,
        ),
    )
    fig.update_yaxes(
        title_text="アグリゲートGEX（億円）",
        gridcolor="#1a1a2a",
        zeroline=False,
        secondary_y=True,
    )

    return fig, net_total, gamma_flip, put_wall, call_wall


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.title("📊 日経225 Gamma Exposure ビジュアライザー")
    st.caption("ディーラーのガンマエクスポージャー分布を可視化 — Dealer GEX Analysis")

    today = date.today()

    with st.sidebar:
        st.header("設定")

        # リアルタイム現値取得
        live_spot = fetch_nikkei_spot()
        if live_spot:
            st.caption(f"🔴 LIVE: ¥{live_spot:,.0f}（1分ごと更新）")
            default_spot = int(live_spot)
        else:
            st.caption("現値を手動入力してください")
            default_spot = 38500

        spot = st.number_input(
            "日経225 現値", min_value=10000, max_value=100000,
            value=default_spot, step=100,
        )
        live_rate = fetch_japan_rate()
        if live_rate:
            st.caption(f"📈 金利: {live_rate*100:.2f}%（自動取得）")
            default_rate = round(live_rate * 100, 2)
        else:
            default_rate = 0.1

        risk_free = st.number_input(
            "無リスク金利（%）", min_value=0.0, max_value=5.0,
            value=default_rate, step=0.05,
        ) / 100

        st.divider()
        oi_threshold = st.slider(
            "OI最小フィルター（枚）",
            min_value=0, max_value=1000, value=50, step=10,
            help="建玉がこの枚数未満のストライクを除外",
        )

        st.divider()
        st.subheader("データソース")
        data_mode = st.radio("モード選択", ["最新データ（自動）", "JPX PDFアップロード", "デモデータ"])

        options_df = pd.DataFrame()

        if data_mode == "デモデータ":
            st.info("合成データでデモ表示します。")
            options_df = generate_demo_data(spot, today)

        elif data_mode == "最新データ（自動）":
            csv_path = Path("data/latest.csv")
            if csv_path.exists():
                options_df = pd.read_csv(csv_path, parse_dates=["expiry"])
                data_date = options_df["date"].iloc[0] if "date" in options_df.columns else "不明"
                st.success(f"最新データ読み込み済み（{data_date}）")
                # days_to_expiry を再計算
                options_df["days_to_expiry"] = (
                    pd.to_datetime(options_df["expiry"]).dt.date.apply(
                        lambda d: (d - today).days
                    )
                )
                options_df = options_df[options_df["days_to_expiry"] > 0]
            else:
                st.warning("data/latest.csv がまだありません。ZIPをアップロードして処理してください。")

        else:
            uploaded = st.file_uploader(
                "JPX日次相場表をアップロード",
                type=["pdf", "zip"],
                help="Daily_Report_OSE_*.zip または siop_dyr_*.pdf を直接",
            )
            if uploaded:
                raw = uploaded.read()
                if uploaded.name.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        pdf_files = [n for n in zf.namelist() if "siop" in n.lower() and n.endswith(".pdf")]
                        if pdf_files:
                            raw_pdf = zf.read(pdf_files[0])
                            st.info(f"使用ファイル: {pdf_files[0]}")
                            with st.spinner("PDFを解析中（IV計算含む）…"):
                                options_df = parse_jpx_pdf(raw_pdf, today, spot, risk_free)
                        else:
                            st.error("ZIP内にsiop_*.pdfが見つかりません。")
                elif uploaded.name.endswith(".pdf"):
                    with st.spinner("PDFを解析中…"):
                        options_df = parse_jpx_pdf(raw, today, spot, risk_free)

    if options_df.empty:
        st.warning("データが読み込まれていません。サイドバーでモードを選択してください。")
        _show_jpx_guide()
        return

    # GEX計算
    gex_df = calculate_gex(options_df, spot, risk_free)

    # 満期フィルター
    expiries = sorted(gex_df["expiry"].unique())
    expiry_labels = ["全満期合算"] + [pd.Timestamp(e).strftime("%Y/%m/%d") for e in expiries]
    selected_label = st.selectbox("満期日フィルター", expiry_labels)
    selected_expiry = "全満期合算" if selected_label == "全満期合算" else expiries[expiry_labels.index(selected_label) - 1]

    # チャート描画
    result = build_gex_chart(gex_df, spot, selected_expiry, oi_threshold)
    fig, net_total, gamma_flip, put_wall, call_wall = result

    # KPIメトリクス
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Net GEX", f"{net_total/1e8:.1f} 億円",
        delta="Long Gamma" if net_total > 0 else "Short Gamma"
    )
    col2.metric("現値", f"{spot:,.0f}")
    col3.metric("ガンマフリップ", f"{gamma_flip:,.0f}" if gamma_flip else "N/A")
    col4.metric("プットウォール", f"{put_wall:,.0f}" if put_wall else "N/A")
    col5.metric("コールウォール", f"{call_wall:,.0f}" if call_wall else "N/A")

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("表示できるデータがありません。OIフィルターを下げるか現値を調整してください。")

    # IVスマイル（実データ時のみ）
    if "settlement" in options_df.columns:
        with st.expander("📈 インプライドボラティリティ スマイル"):
            iv_df = options_df[
                (options_df["strike"] >= spot * 0.75) &
                (options_df["strike"] <= spot * 1.25) &
                (options_df["settlement"] > 0)
            ].copy()
            if not iv_df.empty:
                fig_iv = go.Figure()
                for opt_type, color in [("call", "#e74c3c"), ("put", "#27ae60")]:
                    d = iv_df[iv_df["type"] == opt_type].sort_values("strike")
                    fig_iv.add_trace(go.Scatter(
                        x=d["strike"], y=d["iv"] * 100,
                        mode="lines+markers",
                        name=f"{'コール' if opt_type=='call' else 'プット'}",
                        line=dict(color=color),
                        hovertemplate="Strike: %{x:,.0f}<br>IV: %{y:.1f}%<extra></extra>",
                    ))
                fig_iv.update_layout(
                    title="IVスマイル（清算値より算出）",
                    xaxis_title="行使価格", yaxis_title="IV (%)",
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font_color="#fafafa", height=350,
                    xaxis=dict(gridcolor="#2a2a2a"),
                    yaxis=dict(gridcolor="#2a2a2a"),
                )
                st.plotly_chart(fig_iv, use_container_width=True)

    with st.expander("📖 GEXの読み方"):
        st.markdown("""
**Gamma Exposure (GEX) とは**

オプション市場において、ディーラー（マーケットメーカー）が保有するガンマポジションの総量をストライク別に可視化したものです。

| 指標 | 意味 |
|------|------|
| **コールGEX（赤）** | そのストライクのコールOIによるGEX |
| **プットGEX（緑）** | そのストライクのプットOIによるGEX |
| **アグリゲートGEX（青線）** | ストライクを低い方から累積したNet GEX |
| **ガンマフリップ** | 累積GEXがゼロを交差するストライク。現値がここを下回ると相場の性質が変わりやすい |
| **プットウォール** | プットGEXが最大のストライク＝下値支持として機能しやすい |
| **コールウォール** | コールGEXが最大のストライク＝上値抵抗として機能しやすい |

> OIベースのため日中変化は捕捉できません。他シグナルと組み合わせて参照してください。
        """)

    with st.expander("📋 生データ（ストライク別GEX）"):
        summary = gex_df.groupby("strike").agg(
            call_gex=("call_gex", "sum"),
            put_gex=("put_gex", "sum"),
            net_gex=("gex", "sum"),
        ).reset_index()
        summary["call_億円"] = (summary["call_gex"] / 1e8).round(1)
        summary["put_億円"] = (summary["put_gex"] / 1e8).round(1)
        summary["net_億円"] = (summary["net_gex"] / 1e8).round(1)
        summary = summary.sort_values("net_億円", ascending=False).reset_index(drop=True)
        st.dataframe(summary[["strike", "call_億円", "put_億円", "net_億円"]], use_container_width=True)


def _show_jpx_guide():
    with st.expander("📥 JPXデータの取得方法", expanded=True):
        st.markdown("""
1. [JPX 大阪取引所日報](https://www.jpx.co.jp/markets/statistics-derivatives/daily/index.html) を開く
2. 最新日付の **OSE「概算・精算相場表」ZIP** をダウンロード
3. このアプリのサイドバーからZIPをそのままアップロード
        """)


if __name__ == "__main__":
    main()
