import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from datetime import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# ================== FORMATTERS ==================
def fmt_currency(x):
    if x is None:
        return "N/A"
    x = float(x)
    if x >= 1e12:
        return f"₹{x/1e12:.2f}T"
    elif x >= 1e9:
        return f"₹{x/1e9:.2f}B"
    elif x >= 1e7:
        return f"₹{x/1e7:.2f}Cr"
    elif x >= 1e5:
        return f"₹{x/1e5:.2f}L"
    else:
        return f"₹{x:,.2f}"

def fmt_price(x):
    return "N/A" if x is None else f"₹{x:,.2f}"

def fmt_volume(x):
    if x is None:
        return "N/A"
    x = float(x)
    if x >= 1e7:
        return f"{x/1e7:.2f}Cr"
    elif x >= 1e6:
        return f"{x/1e6:.2f}M"
    elif x >= 1e5:
        return f"{x/1e5:.2f}L"
    else:
        return f"{int(x):,}"

def fmt_count(x):
    if x is None:
        return "N/A"
    x = float(x)
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    elif x >= 1e3:
        return f"{x/1e3:.2f}K"
    else:
        return f"{int(x)}"

def fmt_epoch_date(epoch):
    if epoch is None:
        return "N/A"
    try:
        return dt.fromtimestamp(epoch).strftime("%b %d, %Y")
    except:
        return "N/A"

# ================== APP CONFIG ==================
st.set_page_config(page_title="Quant Stock Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .topbar {
        background-color: #0f172a;
        color: white;
        padding: 14px 20px;
        font-size: 22px;
        font-weight: 600;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="topbar">📈 Quant Stock Dashboard</div>', unsafe_allow_html=True)

st.sidebar.header("Menu")
menu = st.sidebar.radio(
    "Select Option",
    ["Check Stock Analysis", "Compare Two Stocks", "Portfolio Making"]
)

# ================== STOCK DATABASE ==================
stocks_db = {
    "RELIANCE": "Reliance Industries",        # NOT RIL
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "AXISBANK": "Axis Bank",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "ITC": "ITC Limited",
    "HINDUNILVR": "Hindustan Unilever",
    "BHARTIARTL": "Bharti Airtel",
    "ASIANPAINT": "Asian Paints",
    "MARUTI": "Maruti Suzuki",
    "HCLTECH": "HCL Technologies",
    "WIPRO": "Wipro",
    "SUNPHARMA": "Sun Pharmaceutical",
    "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv",
    "ULTRACEMCO": "UltraTech Cement",
    "TITAN": "Titan Company",
    "ONGC": "Oil & Natural Gas Corp",
    "NTPC": "NTPC Limited",
    "POWERGRID": "Power Grid Corp",
    "COALINDIA": "Coal India",
    "JSWSTEEL": "JSW Steel",
    "TATASTEEL": "Tata Steel",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "GRASIM": "Grasim Industries",
    "DIVISLAB": "Divi's Laboratories",
    "DRREDDY": "Dr Reddy's Laboratories",
    "EICHERMOT": "Eicher Motors",
    "HEROMOTOCO": "Hero MotoCorp",
    "HDFCLIFE": "HDFC Life Insurance",
    "SBILIFE": "SBI Life Insurance",
    "INDUSINDBK": "IndusInd Bank",
    "BRITANNIA": "Britannia Industries",
    "CIPLA": "Cipla",
    "TECHM": "Tech Mahindra",
    "APOLLOHOSP": "Apollo Hospitals",
    "TATAMOTORS": "Tata Motors",
    "UPL": "UPL Limited",
    "NESTLEIND": "Nestle India",
    "HAVELLS": "Havells India",
    "PIDILITIND": "Pidilite Industries",
    "SHREECEM": "Shree Cement",
    "BPCL": "Bharat Petroleum",
    "IOC": "Indian Oil Corporation"
}


# ================== MAIN ==================
if menu == "Check Stock Analysis":

    st.markdown("### Time Duration")
    duration = st.radio(
        "", ["1D","5D","1M","6M","YTD","1Y","5Y","MAX","Custom"], horizontal=True
    )

    period, start_date, end_date = "max", None, None
    if duration == "1D": period = "1d"
    elif duration == "5D": period = "5d"
    elif duration == "1M": period = "1mo"
    elif duration == "6M": period = "6mo"
    elif duration == "1Y": period = "1y"
    elif duration == "5Y": period = "5y"
    elif duration == "YTD":
        start_date = datetime.date(datetime.date.today().year, 1, 1)
        end_date = datetime.date.today()
    elif duration == "Custom":
        c1, c2 = st.columns(2)
        with c1: start_date = st.date_input("Start Date")
        with c2: end_date = st.date_input("End Date")

    st.subheader("Select Stock")
    q = st.text_input("Search stocks (name or symbol)")

    df = pd.DataFrame([{"Symbol":k,"Company":v} for k,v in stocks_db.items()])
    if q:
        q = q.lower()
        df = df[df["Symbol"].str.lower().str.contains(q) |
                df["Company"].str.lower().str.contains(q)]

    st.dataframe(df, hide_index=True, use_container_width=True)
    
    symbol = st.selectbox(
        "Select stock",
        df["Symbol"].tolist()
    )


    if symbol:
        ticker = yf.Ticker(symbol + ".NS")
        try:
            data = (ticker.history(start=start_date, end=end_date)
                    if duration in ["YTD","Custom"]
                    else ticker.history(period=period))
            if data.empty or len(data) < 5:
                st.warning("Selected range unavailable. Showing full history.")
                data = ticker.history(period="max")
        except:
            st.error("Error fetching data")
            st.stop()

        info = ticker.info
        st.success(f"{stocks_db[symbol]} ({symbol}.NS)")

        # ================== OVERVIEW ==================
        st.subheader("Overview")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("Open", fmt_price(info.get("open")))
            st.write("High", fmt_price(info.get("dayHigh")))
            st.write("Low", fmt_price(info.get("dayLow")))
            st.write("Market Cap", fmt_currency(info.get("marketCap")))
            st.write("Avg. Volume", fmt_volume(info.get("averageVolume")))

        with c2:
            period_volume = data["Volume"].sum()
            st.write("Volume (Selected Period)", fmt_volume(period_volume))
            st.write("Dividend Yield", info.get("dividendYield","N/A"))
            st.write("Quarterly Dividend", fmt_price(info.get("trailingAnnualDividendRate")))
            st.write("Ex-Dividend Date", fmt_epoch_date(info.get("exDividendDate")))
            st.write("P/E Ratio", info.get("trailingPE","N/A"))

        with c3:
            st.write("52W High", fmt_price(info.get("fiftyTwoWeekHigh")))
            st.write("52W Low", fmt_price(info.get("fiftyTwoWeekLow")))
            st.write("EPS", fmt_price(info.get("trailingEps")))
            st.write("Shares Outstanding", fmt_count(info.get("sharesOutstanding")))
            st.write("Employees", fmt_count(info.get("fullTimeEmployees")))

        # ================== PRO CANDLE CHART ==================
        st.subheader("🕯️ Advanced Candlestick Chart")

        # -------- Indicators --------
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

        # Bollinger Bands
        data["STD"] = data["Close"].rolling(20).std()
        data["Upper"] = data["MA20"] + (2 * data["STD"])
        data["Lower"] = data["MA20"] - (2 * data["STD"])

        # -------- Subplots (Price + Volume) --------
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )

        # -------- Candlestick --------
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red"
            ),
            row=1, col=1
        )

        # -------- Moving Averages --------
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA20"],
                line=dict(color="yellow", width=1.5),
                name="MA20"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA50"],
                line=dict(color="red", width=1.5),
                name="MA50"
            ),
            row=1, col=1
        )

        # -------- Bollinger Bands --------
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Upper"],
                line=dict(color="gray", dash="dot"),
                name="Upper Band"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Lower"],
                line=dict(color="gray", dash="dot"),
                name="Lower Band",
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.1)"
            ),
            row=1, col=1
        )

        # -------- Volume (Bottom panel) --------
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color="lightblue"
            ),
            row=2, col=1
        )

        # -------- Layout --------
        fig.update_layout(
            template="plotly_dark",
            height=750,
            xaxis_rangeslider_visible=False,
            title="Price Action with Indicators",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # ================== CHART INTERPRETATION ==================
        st.markdown("---")
        st.subheader("📚 Indicator Explanation")

        with st.expander("🕯️ Candlestick Chart"):
            st.write("""
            - Each candle represents price movement for a time period.
            - **Green Candle** → Closing price higher than opening (Bullish).
            - **Red Candle** → Closing price lower than opening (Bearish).
            - **Upper wick** → Highest traded price.
            - **Lower wick** → Lowest traded price.
            """)

        with st.expander("📈 Moving Averages (MA20 & MA50)"):
            st.write("""
            - Moving averages smooth price trends.
            - **MA20 (Yellow Line)** → Short-term trend indicator.
            - **MA50 (Red Line)** → Long-term trend indicator.
            - Price above MA → Bullish sentiment.
            - Price below MA → Bearish sentiment.
            """)

        with st.expander("📊 Bollinger Bands"):
            st.write("""
            Bollinger Bands measure market volatility.

            Components:
            - **Middle Band** → 20-day moving average.
            - **Upper Band** → MA + 2 Standard Deviations.
            - **Lower Band** → MA − 2 Standard Deviations.

            Interpretation:
            - Bands widening → High volatility.
            - Bands squeezing → Low volatility (possible breakout).
            - Price touching upper band → Overbought.
            - Price touching lower band → Oversold.
            """)

        with st.expander("📦 Volume"):
            st.write("""
            Volume shows number of shares traded.

            Interpretation:
            - Rising price + high volume → Strong bullish move.
            - Falling price + high volume → Strong bearish move.
            - Low volume → Weak trend / low conviction.
            """)


        # ================== QUANT ==================
        st.subheader("Quantitative Analysis")
        data["Returns"] = data["Close"].pct_change()

        daily_vol = data["Returns"].std()
        annual_vol = daily_vol * np.sqrt(252)
        risk = "Low 🟢" if annual_vol < 0.20 else "Medium 🟡" if annual_vol < 0.35 else "High 🔴"

        # ---- Market Sentiment (Safe) ----
        window = min(20, len(data) // 2)

        if window > 0:
            trend = data["Close"].iloc[-1] - data["Close"].iloc[-window]
            sentiment = "Bullish 📈" if trend > 0 else "Bearish 📉"
        else:
            sentiment = "Neutral ⚪"


        q1,q2,q3 = st.columns(3)
        q1.metric("Daily Volatility", f"{daily_vol:.4f}")
        q2.metric("Annualized Volatility", f"{annual_vol:.2%}")
        q3.metric("Risk Level", risk)

        st.write("Market Sentiment:", sentiment)

        # ================== PRICE RANGE ==================
        st.subheader("Price Range Analysis")
        start_p = data["Close"].iloc[0]
        hi, lo = data["High"].max(), data["Low"].min()
        avg_price = data["Close"].mean()
        range_vol = ((hi - lo) / avg_price) * 100


        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Start Price", fmt_price(start_p))
        r2.metric("Period High", fmt_price(hi))
        r3.metric("Period Low", fmt_price(lo))
        r4.metric("Range Volatility (Avg - Normalized)", f"{range_vol:.2f}%")

elif menu == "Compare Two Stocks":

    st.subheader("📊 Stock Comparison")

    # -------- Select Stocks --------
    compare_symbols = st.multiselect(
        "Select stocks to compare (min 2, max 5)",
        options=list(stocks_db.keys()),
        max_selections=5
    )

    if len(compare_symbols) < 2:
        st.info("Please select at least 2 stocks to compare.")
        st.stop()

    # -------- Time Duration --------
    st.markdown("### Time Duration")

    duration = st.radio(
        "",
        ["1M", "6M", "1Y", "5Y", "MAX"],
        horizontal=True,
        key="compare_duration"
    )

    period_map = {
        "1M": "1mo",
        "6M": "6mo",
        "1Y": "1y",
        "5Y": "5y",
        "MAX": "max"
    }

    period = period_map[duration]

    comparison_rows = []
    price_df = pd.DataFrame()
    start_dates = {}

    # -------- Fetch + Compute Metrics --------
    for symbol in compare_symbols:

        ticker = yf.Ticker(symbol + ".NS")
        data = ticker.history(period=period)

        if data.empty or len(data) < 5:
            continue

        # Save listing start date
        start_dates[symbol] = data.index.min()

        # Price metrics
        start_price = data["Close"].iloc[0]
        end_price = data["Close"].iloc[-1]
        period_return = ((end_price - start_price) / start_price) * 100

        # Volatility
        data["Returns"] = data["Close"].pct_change()
        daily_vol = data["Returns"].std()
        annual_vol = daily_vol * np.sqrt(252)

        # Risk level
        if annual_vol < 0.20:
            risk = "Low 🟢"
        elif annual_vol < 0.35:
            risk = "Medium 🟡"
        else:
            risk = "High 🔴"

        # Price range volatility (avg normalized)
        hi = data["High"].max()
        lo = data["Low"].min()
        avg_price = data["Close"].mean()
        range_vol = ((hi - lo) / avg_price) * 100

        # Store row
        comparison_rows.append({
            "Stock": symbol,
            "Period Return (%)": round(period_return, 2),
            "Annual Volatility (%)": round(annual_vol * 100, 2),
            "Range Volatility (%)": round(range_vol, 2),
            "Risk Level": risk
        })

        # Price data for charts
        price_df[symbol] = data["Close"]

    # -------- Align all stocks to common overlapping date --------
    if not start_dates:
        st.error("Not enough data to compare selected stocks.")
        st.stop()

    common_start = max(start_dates.values())
    price_df = price_df[price_df.index >= common_start]

    if len(price_df) < 5:
        st.warning(
            "⚠️ Selected stocks do not have sufficient overlapping history "
            "for the chosen time range."
        )
        st.stop()

    # -------- Comparison Table --------
    st.subheader("📋 Quantitative Comparison Table")

    comp_df = pd.DataFrame(comparison_rows)

    if comp_df.empty:
        st.error("Not enough data to compare selected stocks.")
        st.stop()

    st.dataframe(comp_df, use_container_width=True)

    # -------- Price Comparison Charts --------
    st.subheader("📈 Price Comparison Charts")

    # Normalized Performance
    st.markdown("#### Normalized Performance (Base = 100)")
    norm_price_df = price_df / price_df.iloc[0] * 100
    st.line_chart(norm_price_df)

    # Actual Prices
    st.markdown("#### Actual Price Movement (₹)")
    st.line_chart(price_df)

    # ================== CANDLESTICK COMPARISON ==================
    st.markdown("---")
    st.subheader("🕯️ Candlestick Comparison")

    import plotly.graph_objects as go

    # Grid layout → 2 charts per row
    cols = st.columns(2)

    for i, symbol in enumerate(compare_symbols):

        ticker = yf.Ticker(symbol + ".NS")
        data = ticker.history(period=period)

        if data.empty or len(data) < 5:
            st.warning(f"Not enough data for {symbol}")
            continue

        # -------- Indicators --------
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

        # -------- Chart --------
        fig = go.Figure()

        # Candles
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red"
        ))

        # Moving averages
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA20"],
            line=dict(color="yellow", width=1),
            name="MA20"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA50"],
            line=dict(color="red", width=1),
            name="MA50"
        ))

        fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
            title=f"{symbol} Candlestick",
            xaxis_rangeslider_visible=False
        )

        # Place chart in grid
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    # ================== CORRELATION HEATMAP ==================
    st.markdown("---")
    st.subheader("📊 Stock Correlation Heatmap")

    # Calculate daily returns
    returns_df = price_df.pct_change().dropna()

    if returns_df.empty:
        st.warning("Not enough data to compute correlation.")
    else:
        corr_matrix = returns_df.corr()

        import plotly.express as px

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix of Selected Stocks"
        )

        fig.update_layout(
            template="plotly_dark",
    height=600
)

    st.plotly_chart(fig, use_container_width=True)



    # -------- Interpretation Help --------
    # ================== INTERPRETATION ==================
    st.markdown("---")
    st.subheader("🧠 How to Read This Dashboard")

    with st.expander("📈 Performance Charts"):
        st.write("""
        **Actual Price Chart**
        - Shows real stock price movement over time.
        - Useful for identifying absolute price levels.

        **Normalized Performance Chart**
        - All stocks start at 100.
        - Enables relative performance comparison.
        - Helps identify outperformers and underperformers.
        """)

    with st.expander("🕯️ Candlestick Charts"):
        st.write("""
        Each candlestick represents price action for a time period.

        - **Green Candle** → Price closed higher than opened.
        - **Red Candle** → Price closed lower than opened.
        - **Upper Wick** → Highest traded price.
        - **Lower Wick** → Lowest traded price.

        Candlesticks reveal market sentiment and volatility.
        """)

    with st.expander("📊 Moving Averages"):
        st.write("""
        Moving averages smooth price fluctuations.

        - **MA20** → Short-term trend indicator.
        - **MA50** → Long-term trend indicator.

        Price above MA → Bullish trend.  
        Price below MA → Bearish trend.
        """)

    with st.expander("📉 Bollinger Bands"):
        st.write("""
        Bollinger Bands measure volatility.

        - Middle Band → 20-day moving average.
        - Upper Band → MA + 2 standard deviations.
        - Lower Band → MA − 2 standard deviations.

        Bands widening → High volatility.  
        Bands squeezing → Low volatility (possible breakout).
        """)

    with st.expander("📦 Volume"):
        st.write("""
        Volume indicates trading activity.

        - High volume + price rise → Strong bullish move.
        - High volume + price fall → Strong bearish move.
        - Low volume → Weak conviction.
        """)

    with st.expander("📊 Correlation Heatmap"):
        st.write("""
        Correlation measures how stocks move relative to each other.

        - +1 → Perfect positive correlation.
        - 0 → No relationship.
        - −1 → Perfect negative correlation.

        Helps identify diversification opportunities.
        """)


elif menu == "Portfolio Making":
    st.subheader("Portfolio Making")
    st.write("Coming soon…")