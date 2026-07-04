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
    """日本の短期金利を複数ソースから取得する"""
    # 試すティッカーのリスト（日本短期金利系）
    tickers = [
        "%5EJP10YB",   # 日本10年国債
        "^N6M.T",      # 日本6ヶ月TB
    ]
    headers = {"User-Agent": "Mozilla/5.0"}

    for ticker in tickers:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d"
            resp = requests.get(url, headers=headers, timeout=5)
            data = resp.json()
            rate = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
            r = float(rate) / 100
            if 0 <= r <= 0.05:  # 0〜5%の範囲なら有効
                return r
        except Exception:
            continue

    # 最終手段：日銀サイトから政策金利を取得
    try:
        resp = requests.get(
            "https://www.boj.or.jp/statistics/money/call/index.htm",
            headers=headers, timeout=5
        )
        import re
        match = re.search(r'(\d+\.\d+)\s*%', resp.text)
        if match:
            r = float(match.group(1)) / 100
            if 0 <= r <= 0.05:
                return r
    except Exception:
        pass

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
    MAX_OI = 500_000  # 50万枚超は誤読データとして除外
    records = []
    for _, row in df.iterrows():
        if row["oi"] > MAX_OI:
            continue  # パーサーの誤読（金額等を誤取得）をスキップ
        T = row["days_to_expiry"] / 365.0
        gamma = bs_gamma(spot, row["strike"], T, r, row["iv"])
        gex = gamma * row["oi"] * MULTIPLIER * spot
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
        filtered = gex_df
    elif isinstance(selected_expiry, list):
        filtered = gex_df[gex_df["expiry"].isin(selected_expiry)]
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

    # アグリゲートGEX（高ストライク→低ストライク方向に累積）
    # Tiger Brokers方式: 右端（高ストライク）から左に向かって積み上げる
    agg["agg_gex"] = agg["gex"].iloc[::-1].cumsum().iloc[::-1]

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

    # ── 背景ゾーン（Tiger風：ガンマフリップで左右にパキッと分割）──
    GREEN_BG = "rgba(0,200,150,0.07)"
    PINK_BG = "rgba(255,92,122,0.07)"
    x_lo = agg["strike"].min() - 125
    x_hi = agg["strike"].max() + 125

    if gamma_flip is not None:
        # フリップの左側の累積GEXの符号でゾーン色を決める
        left_side = agg[agg["strike"] < gamma_flip]
        left_positive = (not left_side.empty) and (left_side["agg_gex"].mean() >= 0)
        left_color = GREEN_BG if left_positive else PINK_BG
        right_color = PINK_BG if left_positive else GREEN_BG
        fig.add_vrect(x0=x_lo, x1=gamma_flip, fillcolor=left_color, layer="below", line_width=0)
        fig.add_vrect(x0=gamma_flip, x1=x_hi, fillcolor=right_color, layer="below", line_width=0)
    else:
        # フリップなし＝全域同一符号
        whole_color = GREEN_BG if agg["agg_gex"].mean() >= 0 else PINK_BG
        fig.add_vrect(x0=x_lo, x1=x_hi, fillcolor=whole_color, layer="below", line_width=0)

    # ── プットGEX バー（緑）──
    fig.add_trace(
        go.Bar(
            x=agg["strike"],
            y=agg["put_gex"] / unit,
            name="プット GEX",
            marker_color="#00C896",
            marker_line_width=0,
            width=110,
            opacity=0.95,
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
            marker_color="#FF5C7A",
            marker_line_width=0,
            width=110,
            opacity=0.95,
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
            line=dict(color="#6C9FFF", width=2.5, shape="spline", smoothing=0.6),
            hovertemplate="Strike: %{x:,.0f}<br>Aggregate GEX: %{y:.1f} 億円<extra></extra>",
        ),
        secondary_y=True,
    )

    # ── ガンマフリップライン（オレンジ実線・上に数値）──
    if gamma_flip:
        fig.add_vline(x=gamma_flip, line_width=2, line_color="#FF8A00")
        fig.add_annotation(
            x=gamma_flip, y=1.06, yref="paper",
            text=f"<b>{gamma_flip:,.0f}</b>",
            showarrow=False,
            font=dict(color="#FF8A00", size=12),
        )

    # ── 現値ライン（グレー破線・上に数値）──
    fig.add_vline(x=spot, line_width=1.5, line_dash="dash", line_color="#6b7280")
    fig.add_annotation(
        x=spot, y=1.06, yref="paper",
        text=f"<b>{spot:,.0f}</b>",
        showarrow=False,
        font=dict(color="#3a4356", size=12),
    )

    # ── コールウォール（矢印付きラベル）──
    cw_y = float(agg.loc[call_wall_idx, "call_gex"]) / unit
    fig.add_annotation(
        x=call_wall, y=cw_y,
        text=f"コールウォール {call_wall:,.0f}",
        showarrow=True,
        arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor="#FF5C7A",
        ax=0, ay=-32,
        font=dict(color="#FF5C7A", size=11),
    )

    # ── プットウォール（矢印付きラベル）──
    pw_y = float(agg.loc[put_wall_idx, "put_gex"]) / unit
    fig.add_annotation(
        x=put_wall, y=pw_y,
        text=f"プットウォール {put_wall:,.0f}",
        showarrow=True,
        arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor="#00C896",
        ax=0, ay=32,
        font=dict(color="#00C896", size=11),
    )

    # ── ゼロライン（破線）──
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(0,0,0,0.3)", secondary_y=False)

    # x軸ティック（範囲に応じて間隔を自動調整 → 目盛りが詰まらない）
    x_min = int(agg["strike"].min())
    x_max = int(agg["strike"].max())
    x_range = x_max - x_min
    tick_step = max(500, int(round(x_range / 12 / 500)) * 500)  # 目盛りは12本前後に
    tick_vals = list(range(round(x_min / tick_step) * tick_step,
                           round(x_max / tick_step) * tick_step + tick_step, tick_step))

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#3a4356",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.30,
            xanchor="center", x=0.5,
            font=dict(size=11, color="#5b6478"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=45, b=80),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#c3cde8",
            font=dict(color="#1a2233", size=12),
        ),
        xaxis=dict(
            title=dict(text="行使価格", font=dict(size=11, color="#5b6478")),
            tickvals=tick_vals,
            ticktext=[f"{v:,}" for v in tick_vals],
            tickangle=-45,
            tickfont=dict(size=11, color="#3a4356"),
            gridcolor="rgba(0,0,0,0.05)",
        ),
        yaxis=dict(
            title=dict(text="GEX（億円）", font=dict(size=11, color="#5b6478")),
            tickfont=dict(size=11, color="#3a4356"),
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
        ),
    )
    fig.update_yaxes(
        title_text="累積GEX（億円）",
        title_font=dict(size=11, color="#4C6FFF"),
        tickfont=dict(size=11, color="#4C6FFF"),
        gridcolor="rgba(76,111,255,0.07)",
        zeroline=False,
        secondary_y=True,
    )

    return fig, net_total, gamma_flip, put_wall, call_wall


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="日経225 GEX",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # スタイリッシュ・カスタムCSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Sans+JP:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans JP', sans-serif;
    }

    /* Streamlitの余計なUIを隠す */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* ヒーロータイトル */
    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C9FFF 0%, #A78BFA 50%, #FF5C7A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: 0.02em;
    }
    .hero-sub {
        color: #8b93a7;
        font-size: 0.8rem;
        margin-top: 2px;
        margin-bottom: 12px;
    }

    /* メトリクスカード：白ベースのソフトカード */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e8f5;
        border-radius: 14px;
        padding: 14px 10px;
        box-shadow: 0 2px 10px rgba(30,50,120,0.07);
        text-align: center;
        transition: box-shadow 0.2s, border-color 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: #b9c6f5;
        box-shadow: 0 4px 16px rgba(30,50,120,0.13);
    }
    [data-testid="stMetric"] label {
        font-size: 0.72rem !important;
        color: #6b7690 !important;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #1a2233 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* アラート/情報ボックス */
    .stAlert {
        border-radius: 12px;
        font-size: 0.9rem;
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* エクスパンダー */
    [data-testid="stExpander"] {
        border: 1px solid #e3e8f5;
        border-radius: 12px;
        background: #fafbfe;
    }

    /* マルチセレクトのタグ */
    [data-baseweb="tag"] {
        background: linear-gradient(90deg, #4C6FFF, #6C9FFF) !important;
        border-radius: 8px !important;
        color: #fff !important;
    }

    /* 区切り線を薄く */
    hr { border-color: #e9edf7 !important; }

    /* スマホ最適化 */
    @media (max-width: 640px) {
        [data-testid="column"] { min-width: calc(33% - 8px) !important; }
        .hero-title { font-size: 1.15rem; }
        [data-testid="stMetricValue"] { font-size: 1.0rem !important; }
        [data-testid="stMetric"] { padding: 10px 6px; }
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero-title">日経225 GEX ダッシュボード</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Dealer Gamma Exposure — JPX公式データ × 毎朝自動更新</div>', unsafe_allow_html=True)

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
            default_rate = 0.75  # 日銀政策金利ベースのデフォルト値

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

    # 満期フィルター（複数選択可）
    expiries = sorted(gex_df["expiry"].unique())
    expiry_label_map = {pd.Timestamp(e).strftime("%Y/%m/%d"): e for e in expiries}
    expiry_options = list(expiry_label_map.keys())

    selected_labels = st.multiselect(
        "満期日フィルター（複数選択可・未選択=全合算）",
        options=expiry_options,
        default=[],
        help="何も選ばないと全満期合算。複数選ぶと選択分を合算して表示。"
    )

    if len(selected_labels) == 0:
        selected_expiry = "全満期合算"
    else:
        selected_expiry = [expiry_label_map[l] for l in selected_labels]

    # チャート描画
    result = build_gex_chart(gex_df, spot, selected_expiry, oi_threshold)
    fig, net_total, gamma_flip, put_wall, call_wall = result

    # KPIメトリクス（スマホ対応：2行に分ける）
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 現値", f"{spot:,.0f}")
    col2.metric("Net GEX", f"{net_total/1e8:.1f} 億円",
        delta="▲ Long Gamma" if net_total > 0 else "▼ Short Gamma")
    col3.metric("⚡ ガンマフリップ", f"{gamma_flip:,.0f}" if gamma_flip else "N/A")

    col4, col5 = st.columns(2)
    col4.metric("🟢 プットウォール", f"{put_wall:,.0f}" if put_wall else "N/A")
    col5.metric("🔴 コールウォール", f"{call_wall:,.0f}" if call_wall else "N/A")

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
                for opt_type, color in [("call", "#FF5C7A"), ("put", "#00C896")]:
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
                    plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#3a4356", height=350,
                    xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                )
                st.plotly_chart(fig_iv, use_container_width=True)

    # ── 今日の詳細分析（データ連動）────────────────────────────────────────────
    st.divider()
    st.subheader("📊 MM（マーケットメーカー）分析レポート")

    is_long_gamma = net_total > 0
    put_dist = ((spot - put_wall) / spot * 100) if put_wall else None
    call_dist = ((call_wall - spot) / spot * 100) if call_wall else None

    # ガンマフリップとの位置関係
    if gamma_flip:
        flip_alert = spot < gamma_flip
        flip_dist = abs(spot - gamma_flip) / spot * 100
    else:
        flip_alert = False
        flip_dist = None

    # ── 総合判断バナー ──
    if is_long_gamma and not flip_alert:
        overall_icon = "🟢"
        overall_label = "ポジティブガンマ環境（Long Gamma）"
        overall_color = "success"
    else:
        overall_icon = "🔴"
        overall_label = "ネガティブガンマ環境（Short Gamma）"
        overall_color = "error"

    # ── ① 青線の状態（最重要）──
    if is_long_gamma:
        st.success(f"""
### 🟢 青線はゼロ以上 → ポジティブガンマ環境

複数の業者が**全員、売りヘッジ**をかけている状態です。
相場が上がれば業者は売り、下がれば買う。この動きが機械的・強制的に入り続けます。
""")
    else:
        st.error(f"""
### 🔴 青線はゼロ以下 → ネガティブガンマ環境

業者が**全員、順張りヘッジ**をかけている状態です。
相場が上がれば業者も買い、下がれば売る。動いた方向にさらに加速しやすい局面です。
""")

    # ── ② 現在地の状況判断 ──
    st.markdown("#### 📌 現在地の分析")

    if is_long_gamma and not flip_alert:
        # ケース1: ガンマフリップとコールウォールの間（最確ゾーン）
        if gamma_flip and call_wall:
            st.info(f"""
**🎯 現値（{spot:,.0f}）はガンマフリップ（{gamma_flip:,.0f}）〜コールウォール（{call_wall:,.0f}）の間にいます。**

このゾーンでは業者が全員売りヘッジをかけているため、相場の動きが最も読みやすい状態です。

- 上がれば業者の売りで抑えられ、コールウォール（{call_wall:,.0f}）が天井として機能しやすい
- 下がれば業者の買いで支えられ、ガンマフリップ（{gamma_flip:,.0f}）がサポートになりやすい

**⚠️ ガンマフリップ（{gamma_flip:,.0f}）を割ったら環境が一変するため、撤退の目安にしてください。**
""")
        # ケース2: コールウォールの上（ポジティブガンマだがウォール上）
        elif call_wall and spot > call_wall:
            st.warning(f"""
**現値（{spot:,.0f}）はコールウォール（{call_wall:,.0f}）の上にいます。**

青線はまだプラスのため業者全体では売りヘッジ継続中です。
ただし、コールウォールを売った業者（一部）は{call_wall:,.0f}を超えた時点で買いに転換しています。
全体としては売り超ですが、局所的な買いが混在している状態です。
""")
        else:
            st.info(f"""
**現値（{spot:,.0f}）はポジティブガンマ環境にいます。**
業者の売りヘッジが相場を支えています。
""")

    elif flip_alert:
        # ケース3: ガンマフリップの下（ネガティブガンマ）
        st.error(f"""
**⚠️ 現値（{spot:,.0f}）はガンマフリップ（{gamma_flip:,.0f}）の下にいます。**

業者が順張りヘッジに転じており、下落が加速しやすい状態です。
{f"プットウォール（{put_wall:,.0f}）が下値の目安ですが、割り込むとさらに売りが加速する可能性があります。" if put_wall else ""}

ガンマフリップ（{gamma_flip:,.0f}）を上抜け回復できるかが焦点です。
""")
    else:
        st.info(f"現値（{spot:,.0f}）の状況を確認中です。")

    # ── ③ キーレベル一覧 ──
    st.markdown("#### 🗺️ キーレベル")
    col1, col2, col3 = st.columns(3)
    with col1:
        if gamma_flip:
            st.metric("⚡ ガンマフリップ", f"{gamma_flip:,.0f}",
                delta=f"現値より{flip_dist:.1f}%{'下' if flip_alert else '下'}",
                delta_color="inverse")
    with col2:
        if put_wall and put_dist is not None:
            st.metric("🟢 プットウォール（下値の壁）", f"{put_wall:,.0f}",
                delta=f"現値より{put_dist:.1f}%下", delta_color="off")
    with col3:
        if call_wall and call_dist is not None:
            st.metric("🔴 コールウォール（上値の壁）", f"{call_wall:,.0f}",
                delta=f"現値より{call_dist:.1f}%上", delta_color="off")

    st.caption("⚠️ OIベースのため日中変化は捕捉できません。他のシグナルと組み合わせてご利用ください。")

    # ── 固定解説（折りたたみ）────────────────────────────────────────────────────
    with st.expander("📖 GEXとは？（基礎知識）"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**💡 ひとことで言うと**
「ディーラーが相場の安全弁になってくれているかどうか」を見るツールです。

ディーラー（証券会社）はオプションを売った分のリスクを消すため、日経225先物を自動売買します。

- コールが多い → 相場が動くたびに反対売買 → **安定する**
- プットが多い → 相場と同じ方向に動く → **荒れやすくなる**
            """)
        with c2:
            st.markdown("""
**📊 各指標の見方**

| 指標 | 意味 |
|------|------|
| 🔴 赤バー | コールOIが多いストライク |
| 🟢 緑バー | プットOIが多いストライク |
| 🔵 青線 | プラス＝安定、マイナス＝荒れやすい |
| ⚡ ガンマフリップ | ここを下抜けると急落しやすい |
| 🟢 プットウォール | 下値の壁 |
| 🔴 コールウォール | 上値の壁 |
            """)

    with st.expander("📖 GEXの読み方（詳細）"):
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
