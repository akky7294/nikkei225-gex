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
from datetime import date
import zipfile

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
    """Black-Scholes オプション価格"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes ガンマ"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def implied_vol(S, K, T, r, market_price, option_type="call", default=0.20):
    """清算値からインプライドボラティリティを逆算（brentq法）"""
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
        # 異常値を除外
        if iv < 0.01 or iv > 5.0:
            return default
        return iv
    except Exception:
        return default


# ─── GEX計算 ─────────────────────────────────────────────────────────────────

def calculate_gex(df: pd.DataFrame, spot: float, r: float = 0.001) -> pd.DataFrame:
    """
    オプションチェーンDFからストライク別GEXを計算する
    GEX = Gamma × OI × 乗数 × Spot²
    Call: +GEX（ディーラーLong Gamma）
    Put:  -GEX（ディーラーShort Gamma）
    """
    MULTIPLIER = 1000  # 日経225オプション乗数

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
            "gamma": gamma,
            "oi": row["oi"],
            "iv": row["iv"],
        })

    return pd.DataFrame(records)


# ─── JPX PDF パーサー ─────────────────────────────────────────────────────────

def parse_jpx_pdf(raw: bytes, today: date, spot: float, r: float) -> pd.DataFrame:
    """
    JPX日次相場表PDF（siop_dyr_YYYYMMDD.pdf）をパースする
    清算値カラムからIVを逆算する
    """
    if not PDF_AVAILABLE:
        st.error("pdfplumberが必要です。")
        return pd.DataFrame()

    row_pattern = re.compile(
        r'(20\d{4})\s+(\d{2}\.\d{2})\s+([\d,]+)\s+\d{6,12}(.*)'
    )

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

                # OI（最終トークン）
                oi = 0
                for token in reversed(rest):
                    clean = token.replace(",", "")
                    if re.fullmatch(r'\d+', clean):
                        oi = int(clean)
                        break
                if oi == 0:
                    continue

                # 清算値（後ろから3番目）
                settlement = 0.0
                if len(rest) >= 3:
                    seisan_str = rest[-3].replace(",", "")
                    try:
                        v = float(seisan_str)
                        if v > 0:
                            settlement = v
                    except ValueError:
                        pass

                # 満期日
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

                # IV計算
                T = days / 365.0
                if settlement > 0:
                    iv = implied_vol(spot, strike, T, r, settlement, current_type)
                else:
                    iv = 0.20

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
        f"  |  IV実測: {n_iv}行 / IV推定(20%): {len(df)-n_iv}行"
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
        T = days / 365
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


# ─── チャート描画 ─────────────────────────────────────────────────────────────

def build_gex_chart(gex_df: pd.DataFrame, spot: float, selected_expiry, oi_threshold: int):
    """GEXバーチャートを構築する"""
    if selected_expiry == "全満期合算":
        plot_df = gex_df.groupby("strike", as_index=False).agg(
            gex=("gex", "sum"), oi=("oi", "sum")
        )
    else:
        filtered = gex_df[gex_df["expiry"] == selected_expiry]
        plot_df = filtered.groupby("strike", as_index=False).agg(
            gex=("gex", "sum"), oi=("oi", "sum")
        )

    # 現値±25%の範囲 + OIフィルター
    plot_df = plot_df[
        (plot_df["strike"] >= spot * 0.75) &
        (plot_df["strike"] <= spot * 1.25) &
        (plot_df["oi"] >= oi_threshold)
    ].sort_values("strike")

    if plot_df.empty:
        return None, 0, None

    net_total = plot_df["gex"].sum()
    gamma_flip = _find_gamma_flip(plot_df)

    # GEXを億円単位に変換
    y = plot_df["gex"] / 1e8
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in y]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["strike"],
        y=y,
        marker_color=colors,
        name="Net GEX",
        hovertemplate=(
            "<b>Strike: %{x:,.0f}</b><br>"
            "GEX: %{y:.1f} 億円<br>"
            "<extra></extra>"
        ),
    ))

    # 現値ライン
    fig.add_vline(
        x=spot, line_width=2.5, line_dash="solid", line_color="#f39c12",
        annotation_text=f"現値 {spot:,.0f}",
        annotation_font_color="#f39c12",
        annotation_position="top right",
    )

    # Gamma Flip ライン
    if gamma_flip is not None:
        fig.add_vline(
            x=gamma_flip, line_width=1.5, line_dash="dash", line_color="#9b59b6",
            annotation_text=f"Gamma Flip {gamma_flip:,.0f}",
            annotation_font_color="#9b59b6",
            annotation_position="top left",
        )

    # ゼロライン強調
    fig.add_hline(y=0, line_width=1, line_color="#555")

    # x軸ティックを500刻みに
    x_min = int(plot_df["strike"].min())
    x_max = int(plot_df["strike"].max())
    tick_vals = list(range(
        round(x_min / 500) * 500,
        round(x_max / 500) * 500 + 500,
        500
    ))

    fig.update_layout(
        title=dict(
            text=f"日経225 Gamma Exposure  |  現値: {spot:,.0f}  |  Net GEX: {net_total/1e8:.1f} 億円",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="行使価格",
            tickvals=tick_vals,
            ticktext=[f"{v:,}" for v in tick_vals],
            tickangle=-45,
            gridcolor="#2a2a2a",
        ),
        yaxis=dict(
            title="Gamma Exposure（億円）",
            gridcolor="#2a2a2a",
            zeroline=False,
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=540,
        showlegend=False,
        bargap=0.15,
        margin=dict(b=80),
    )

    return fig, net_total, gamma_flip


def _find_gamma_flip(plot_df: pd.DataFrame):
    """累積GEXがゼロを交差するストライクを返す"""
    sorted_df = plot_df.sort_values("strike")
    cumsum = sorted_df["gex"].cumsum().values
    strikes = sorted_df["strike"].values
    for i in range(1, len(cumsum)):
        if cumsum[i - 1] * cumsum[i] < 0:
            return int(strikes[i])
    return None


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.title("📊 日経225 Gamma Exposure ビジュアライザー")
    st.caption("ディーラーのガンマエクスポージャー分布を可視化 — Dealer GEX Analysis")

    today = date.today()

    with st.sidebar:
        st.header("設定")
        spot = st.number_input(
            "日経225 現値", min_value=10000, max_value=60000,
            value=38500, step=100,
        )
        risk_free = st.number_input(
            "無リスク金利（%）", min_value=0.0, max_value=5.0,
            value=0.1, step=0.05,
        ) / 100

        st.divider()
        oi_threshold = st.slider(
            "OI最小フィルター（枚）",
            min_value=0, max_value=1000, value=50, step=10,
            help="建玉がこの枚数未満のストライクを除外",
        )

        st.divider()
        st.subheader("データソース")
        data_mode = st.radio("モード選択", ["デモデータ", "JPX PDFアップロード"])

        options_df = pd.DataFrame()

        if data_mode == "デモデータ":
            st.info("合成データでデモ表示します。")
            options_df = generate_demo_data(spot, today)
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

    # チャート
    result = build_gex_chart(gex_df, spot, selected_expiry, oi_threshold)
    fig, net_total, gamma_flip = result

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Net GEX", f"{net_total/1e8:.1f} 億円",
        delta="正 = ディーラー Long Gamma" if net_total > 0 else "負 = ディーラー Short Gamma"
    )
    col2.metric("現値", f"{spot:,.0f}")
    col3.metric("Gamma Flip", f"{gamma_flip:,.0f}" if gamma_flip else "N/A")
    col4.metric("対象満期数", len(expiries))

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("表示できるデータがありません。OIフィルターを下げるか、現値を調整してください。")

    # IV分布（実データ時のみ）
    if "settlement" in options_df.columns:
        with st.expander("📈 インプライドボラティリティ分布（スマイル）"):
            iv_df = options_df[
                (options_df["strike"] >= spot * 0.75) &
                (options_df["strike"] <= spot * 1.25) &
                (options_df["settlement"] > 0)
            ].copy()
            if not iv_df.empty:
                fig_iv = go.Figure()
                for opt_type, color in [("call", "#2ecc71"), ("put", "#e74c3c")]:
                    d = iv_df[iv_df["type"] == opt_type].sort_values("strike")
                    fig_iv.add_trace(go.Scatter(
                        x=d["strike"], y=d["iv"] * 100,
                        mode="lines+markers", name=opt_type.capitalize(),
                        line=dict(color=color),
                        hovertemplate="Strike: %{x:,.0f}<br>IV: %{y:.1f}%<extra></extra>",
                    ))
                fig_iv.update_layout(
                    title="IVスマイル（清算値より算出）",
                    xaxis_title="行使価格", yaxis_title="IV (%)",
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font_color="#fafafa", height=380,
                    xaxis=dict(gridcolor="#2a2a2a"),
                    yaxis=dict(gridcolor="#2a2a2a"),
                )
                st.plotly_chart(fig_iv, use_container_width=True)

    with st.expander("📖 GEXの読み方"):
        st.markdown("""
**Gamma Exposure (GEX) とは**

オプション市場において、ディーラー（マーケットメーカー）が保有するガンマポジションの総量をストライク別に可視化したものです。

| GEX | 意味 | 相場への影響 |
|-----|------|------------|
| **正（緑）** | ディーラーが Long Gamma | 現値の変動を**抑制**（戻り売り・押し目買い） |
| **負（赤）** | ディーラーが Short Gamma | 現値の変動を**増幅**（トレンドフォロー） |

**Gamma Flip Point（紫線）**
正負が逆転するストライク。現値がここを下回るとボラティリティ特性が変わる可能性があります。

> OIベースのため日中変化は捕捉できません。他シグナルと組み合わせて参照してください。
        """)

    with st.expander("📋 生データ（ストライク別Net GEX）"):
        summary = gex_df.groupby("strike")["gex"].sum().reset_index()
        summary["gex_億円"] = (summary["gex"] / 1e8).round(2)
        summary = summary.sort_values("gex_億円", ascending=False).reset_index(drop=True)
        st.dataframe(summary[["strike", "gex_億円"]], use_container_width=True)


def _show_jpx_guide():
    with st.expander("📥 JPXデータの取得方法", expanded=True):
        st.markdown("""
1. [JPX 大阪取引所日報](https://www.jpx.co.jp/markets/statistics-derivatives/daily/index.html) を開く
2. 最新日付の **OSE「概算・精算相場表」ZIP** をダウンロード
3. このアプリのサイドバーからZIPをそのままアップロード
        """)


if __name__ == "__main__":
    main()
