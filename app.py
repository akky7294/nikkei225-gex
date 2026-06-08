"""
Nikkei 225 Options Gamma Exposure (GEX) Visualizer
JPX公式CSVデータを使用してガンマエクスポージャーを可視化する
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import io
import re
import requests
from datetime import datetime, date
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

# ─── Black-Scholes ガンマ計算 ───────────────────────────────────────────────

def bs_gamma(S, K, T, r, sigma):
    """Black-Scholesモデルによるガンマ計算"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def calculate_gex(df: pd.DataFrame, spot: float, r: float = 0.001) -> pd.DataFrame:
    """
    オプションチェーンDFからストライク別GEXを計算する

    GEX = Gamma × OI × 乗数 × Spot²
    ディーラーポジション仮定:
      Call売り手=ディーラー → GEX符号を反転 → 正
      Put売り手=ディーラー  → GEX符号そのまま → 負
    市場慣行: dealers are net short options → Call: +GEX, Put: -GEX
    """
    MULTIPLIER = 1000  # 日経225オプション: 1枚 = 指数 × ¥1,000

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

    result = pd.DataFrame(records)
    return result


# ─── JPXデータ読み込み ──────────────────────────────────────────────────────

def parse_jpx_pdf(raw: bytes, today: date) -> pd.DataFrame:
    """
    JPX日次相場表PDF（siop_dyr_YYYYMMDD.pdf）をパースしてオプションチェーンDFを返す
    ページのテキストから PutOptions / CallOptions のセクションを検出し、
    各行の 限月・行使価格・建玉 を正規表現で抽出する
    """
    if not PDF_AVAILABLE:
        st.error("pdfplumberが必要です。requirements.txtにpdfplumberを追加してください。")
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
                contract_ym = m.group(1)   # e.g. 202606
                exp_md = m.group(2)         # e.g. 06.11
                strike_str = m.group(3).replace(",", "")
                rest = m.group(4).strip().split()

                try:
                    strike = int(strike_str)
                except ValueError:
                    continue

                # 建玉(OI)は最後の数値フィールド
                oi = 0
                for token in reversed(rest):
                    clean = token.replace(",", "")
                    if re.fullmatch(r'\d+', clean):
                        oi = int(clean)
                        break

                if oi == 0:
                    continue

                # 満期日を組み立て: contract_ym=202606, exp_md=06.11 → 2026-06-11
                year = int(contract_ym[:4])
                exp_month = int(exp_md.split(".")[0])
                exp_day = int(exp_md.split(".")[1])
                try:
                    expiry = date(year, exp_month, exp_day)
                except ValueError:
                    continue

                days = (expiry - today).days
                if days <= 0:
                    continue

                records.append({
                    "strike": strike,
                    "expiry": pd.Timestamp(expiry),
                    "type": current_type,
                    "oi": oi,
                    "iv": 0.20,
                    "days_to_expiry": days,
                })

    if not records:
        st.error("PDFからデータを抽出できませんでした。siop_dyr_*.pdf ファイルか確認してください。")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    st.success(f"PDF読み込み完了: {len(df)}行（Call:{len(df[df.type=='call'])} / Put:{len(df[df.type=='put'])}）")
    return df


def parse_jpx_csv(raw: bytes) -> pd.DataFrame:
    """
    JPXの日次オプション取引状況CSVをパースする
    フォーマット: 大証が公開している derivatives market data CSV
    列: 限月, 種別(C/P), 権利行使価格, 出来高, 建玉, 清算値, IV等
    """
    try:
        text = raw.decode("cp932", errors="replace")
        df = pd.read_csv(io.StringIO(text), skiprows=0)
        return df
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return pd.DataFrame()


def normalize_dataframe(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """
    アップロードされたCSVを内部形式に正規化する
    内部形式: strike, expiry, type(call/put), oi, iv, days_to_expiry
    """
    # カラム名を小文字・スペース除去で正規化
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # よくあるJPX形式のマッピング（実際のカラム名に合わせて調整）
    col_map = {
        "権利行使価格": "strike",
        "strike_price": "strike",
        "strike": "strike",
        "限月": "expiry",
        "expiry": "expiry",
        "maturity": "expiry",
        "建玉": "oi",
        "open_interest": "oi",
        "oi": "oi",
        "iv": "iv",
        "implied_volatility": "iv",
        "インプライドボラティリティ": "iv",
        "種別": "type_raw",
        "call/put": "type_raw",
        "cp": "type_raw",
        "type": "type_raw",
    }

    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)

    required = ["strike", "expiry", "oi"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"必要なカラムが見つかりません: {missing}\n実際のカラム: {list(df.columns)}")
        return pd.DataFrame()

    # コール/プット判定
    if "type_raw" in df.columns:
        df["type"] = df["type_raw"].astype(str).str.upper().map(
            lambda x: "call" if x in ["C", "CALL", "コール"] else "put"
        )
    else:
        # カラムがない場合はユーザーに選択させる
        st.warning("コール/プット区別カラムが見つかりません。")
        return pd.DataFrame()

    # IVが無い場合は20%と仮定
    if "iv" not in df.columns:
        df["iv"] = 0.20
    else:
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce").fillna(0.20)
        # パーセント表記 (例: 20.5) → 小数 (0.205)
        if df["iv"].mean() > 1.0:
            df["iv"] = df["iv"] / 100.0

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)

    # 満期日をdatetime化
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    df["days_to_expiry"] = (df["expiry"] - pd.Timestamp(today)).dt.days
    df = df[df["days_to_expiry"] > 0]  # 期限切れ除外

    return df[["strike", "expiry", "type", "oi", "iv", "days_to_expiry"]].dropna()


# ─── デモデータ生成 ──────────────────────────────────────────────────────────

def generate_demo_data(spot: float, today: date) -> pd.DataFrame:
    """デモ用の合成オプションチェーンデータを生成する"""
    np.random.seed(42)
    strikes = np.arange(
        round(spot * 0.85 / 500) * 500,
        round(spot * 1.15 / 500) * 500 + 500,
        500,
    )
    expiries_days = [14, 42, 77]  # 2週間、6週間、11週間後

    rows = []
    for days in expiries_days:
        expiry = pd.Timestamp(today) + pd.Timedelta(days=days)
        T = days / 365
        atm_iv = 0.20
        for strike in strikes:
            moneyness = strike / spot
            # スマイル: OTMほどIV高め
            skew = 0.05 * abs(moneyness - 1.0) + 0.02 * max(0, 1.0 - moneyness)
            iv = atm_iv + skew

            # OI: ATM付近に集中、プットは低ストライクに多い
            call_oi_base = np.exp(-3 * max(0, moneyness - 1.0)**2) * 5000
            put_oi_base = np.exp(-3 * max(0, 1.0 - moneyness)**2) * 5000
            call_oi = int(call_oi_base * np.random.uniform(0.7, 1.3))
            put_oi = int(put_oi_base * np.random.uniform(0.7, 1.3))

            rows.append({"strike": strike, "expiry": expiry, "type": "call",
                         "oi": call_oi, "iv": iv, "days_to_expiry": days})
            rows.append({"strike": strike, "expiry": expiry, "type": "put",
                         "oi": put_oi, "iv": iv, "days_to_expiry": days})

    return pd.DataFrame(rows)


# ─── チャート描画 ────────────────────────────────────────────────────────────

def build_gex_chart(gex_df: pd.DataFrame, spot: float, selected_expiry=None):
    """GEXバーチャートを構築する"""
    if selected_expiry == "全満期合算":
        plot_df = gex_df.groupby("strike", as_index=False)["gex"].sum()
    else:
        plot_df = gex_df[gex_df["expiry"] == selected_expiry].groupby(
            "strike", as_index=False
        )["gex"].sum()

    plot_df = plot_df.sort_values("strike")
    plot_df["color"] = plot_df["gex"].apply(lambda x: "#2ecc71" if x > 0 else "#e74c3c")

    # Gamma flip point (ゼロ付近で符号が変わるストライク)
    net_total = plot_df["gex"].sum()
    gamma_flip = _find_gamma_flip(plot_df)

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "bar"}]],
    )

    fig.add_trace(
        go.Bar(
            x=plot_df["strike"],
            y=plot_df["gex"] / 1e8,  # 億円単位
            marker_color=plot_df["color"],
            name="Net GEX",
            hovertemplate=(
                "<b>Strike: %{x:,.0f}</b><br>"
                "GEX: %{y:.2f} 億円<br>"
                "<extra></extra>"
            ),
        )
    )

    # 現値ライン
    fig.add_vline(
        x=spot, line_width=2, line_dash="solid", line_color="#f39c12",
        annotation_text=f"現値 {spot:,.0f}", annotation_position="top right",
    )

    # Gamma Flip ライン
    if gamma_flip is not None:
        fig.add_vline(
            x=gamma_flip, line_width=1.5, line_dash="dash", line_color="#9b59b6",
            annotation_text=f"Gamma Flip {gamma_flip:,.0f}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(
            text=f"日経225 Gamma Exposure  |  現値: {spot:,.0f}  |  Net GEX: {net_total/1e8:.1f} 億円",
            font=dict(size=16),
        ),
        xaxis_title="行使価格",
        yaxis_title="Gamma Exposure（億円）",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a", zeroline=True, zerolinecolor="#555"),
        height=520,
        showlegend=False,
        bargap=0.1,
    )

    return fig, net_total, gamma_flip


def _find_gamma_flip(plot_df: pd.DataFrame):
    """累積GEXがゼロを交差する行使価格を返す"""
    sorted_df = plot_df.sort_values("strike")
    cumsum = sorted_df["gex"].cumsum()
    sign_changes = cumsum[cumsum.diff().fillna(0) != 0]
    flips = []
    for i in range(1, len(cumsum)):
        if cumsum.iloc[i - 1] * cumsum.iloc[i] < 0:
            flips.append(sorted_df["strike"].iloc[i])
    return flips[0] if flips else None


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.title("📊 日経225 Gamma Exposure ビジュアライザー")
    st.caption("ディーラーのガンマエクスポージャー分布を可視化 — Dealer GEX Analysis")

    today = date.today()

    # サイドバー
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
        st.subheader("データソース")
        data_mode = st.radio(
            "モード選択",
            ["デモデータ", "CSVアップロード"],
            help="本番ではJPX公式CSVをアップロードしてください",
        )

        options_df = pd.DataFrame()

        if data_mode == "デモデータ":
            st.info("合成データでデモ表示します。")
            options_df = generate_demo_data(spot, today)

        else:
            uploaded = st.file_uploader(
                "JPX日次相場表をアップロード",
                type=["pdf", "zip"],
                help="JPX公式の日次相場表ZIP（siop_dyr_*.pdf を含む）またはPDFを直接",
            )
            if uploaded:
                raw = uploaded.read()
                if uploaded.name.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        pdf_files = [n for n in zf.namelist() if "siop" in n.lower() and n.endswith(".pdf")]
                        if pdf_files:
                            raw_pdf = zf.read(pdf_files[0])
                            st.info(f"使用ファイル: {pdf_files[0]}")
                            options_df = parse_jpx_pdf(raw_pdf, today)
                        else:
                            st.error("ZIP内にsiop_*.pdfが見つかりません。")
                elif uploaded.name.endswith(".pdf"):
                    options_df = parse_jpx_pdf(raw, today)

    if options_df.empty:
        st.warning("データが読み込まれていません。サイドバーでデモデータを選択するかCSVをアップロードしてください。")
        _show_jpx_guide()
        return

    # GEX計算
    gex_df = calculate_gex(options_df, spot, risk_free)

    # 満期フィルター
    expiries = sorted(gex_df["expiry"].unique())
    expiry_labels = ["全満期合算"] + [pd.Timestamp(e).strftime("%Y/%m/%d") for e in expiries]
    selected_label = st.selectbox("満期日フィルター", expiry_labels)

    if selected_label == "全満期合算":
        selected_expiry = "全満期合算"
    else:
        selected_expiry = expiries[expiry_labels.index(selected_label) - 1]

    # チャート表示
    fig, net_total, gamma_flip = build_gex_chart(gex_df, spot, selected_expiry)

    # KPIメトリクス
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net GEX", f"{net_total/1e8:.1f} 億円",
                delta="正 = ディーラー Long Gamma" if net_total > 0 else "負 = ディーラー Short Gamma")
    col2.metric("現値", f"{spot:,.0f}")
    col3.metric("Gamma Flip", f"{gamma_flip:,.0f}" if gamma_flip else "N/A")
    col4.metric("対象満期数", len(expiries))

    st.plotly_chart(fig, use_container_width=True)

    # GEX解釈
    with st.expander("📖 GEXの読み方"):
        st.markdown("""
**Gamma Exposure (GEX) とは**

オプション市場において、ディーラー（マーケットメーカー）が保有するガンマポジションの総量を、
ストライク別に可視化したものです。

| GEX | 意味 | 相場への影響 |
|-----|------|------------|
| **正（緑）** | ディーラーが Long Gamma | 現値の変動を**抑制**（戻り売り・押し目買い） |
| **負（赤）** | ディーラーが Short Gamma | 現値の変動を**増幅**（トレンドフォロー） |

**Gamma Flip Point（紫線）**
正負が逆転するストライク。現値がここを下回ると市場のボラティリティ特性が変わる可能性があります。

**注意事項**
- ディーラーポジションは直接観測できないため、慣行的に「ディーラー＝オプション売り手」と仮定します
- OIベースのため、日中の変化は捕捉できません（EODデータの限界）
- 単独指標としてではなく、他のシグナルと組み合わせて参照してください
        """)

    # データテーブル
    with st.expander("📋 生データ（ストライク別Net GEX）"):
        summary = gex_df.groupby("strike")["gex"].sum().reset_index()
        summary["gex_億円"] = (summary["gex"] / 1e8).round(2)
        summary = summary.sort_values("strike").reset_index(drop=True)
        st.dataframe(summary[["strike", "gex_億円"]], use_container_width=True)


def _show_jpx_guide():
    with st.expander("📥 JPXデータの取得方法"):
        st.markdown("""
**日本取引所グループ（JPX）公式データ**

1. [JPX マーケットデータ](https://www.jpx.co.jp/markets/statistics-derivatives/daily/index.html) にアクセス
2. 「日次」→「オプション取引状況」からCSVをダウンロード
3. このアプリにアップロード

**必要なカラム（最低限）:**
- 行使価格（strike price）
- 限月（expiry / maturity）
- 建玉（open interest）
- コール/プット区別
- IV（任意 — ない場合は20%で代替）

**代替データソース:**
- Quick（有料）
- Bloomberg（有料）
- SBI/楽天証券画面からの手動取得
        """)


if __name__ == "__main__":
    main()
