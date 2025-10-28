import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf
import ta
from prophet import Prophet
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Shikher's Financial Analysis App - Comprehensive Financial Analysis and Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }

    .sub-header {
        font-size: 1.25rem;
        font-weight: 400;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: 0.01em;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
        letter-spacing: -0.01em;
    }

    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .recommendation-buy {
        color: #059669;
        font-weight: 700;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #86efac;
    }

    .recommendation-sell {
        color: #dc2626;
        font-weight: 700;
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #fca5a5;
    }

    .recommendation-hold {
        color: #d97706;
        font-weight: 700;
        background: linear-gradient(135deg, #fffbeb 0%, #fde68a 100%);
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #fcd34d;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #64748b;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        color: #334155;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .dataframe th {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #374151;
        font-weight: 600;
        padding: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .dataframe td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #f1f5f9;
    }

    /* Chart styling */
    .js-plotly-plot {
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }

    /* Success/Warning/Error styling */
    .element-container .stAlert {
        border-radius: 0.75rem;
        border: none;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }

        .metric-card {
            padding: 1rem;
        }

        .metric-value {
            font-size: 1.5rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Data fetching functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_financial_data(ticker, years=5):
    """Fetch financial data from multiple sources"""
    try:
        from data_fetching_new import data_fetcher
        from utils import detect_market, get_currency_symbol

        # Detect market and get currency
        market = detect_market(ticker)
        currency = get_currency_symbol(market)

        # Fetch financial statements
        balance_sheet, income_stmt, cash_flow = data_fetcher.fetch_financial_data(ticker, years)

        # Get historical data
        hist_data = data_fetcher.fetch_historical_prices(ticker, f"{years}y")

        # Get current price
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice', stock.history(period='1d')['Close'].iloc[-1] if hist_data is not None and not hist_data.empty else None)

        return balance_sheet, income_stmt, cash_flow, hist_data, current_price, market, currency
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None, None, None, None, None, None

# Financial ratios calculation
def calculate_ratios(balance_sheet, income_stmt, current_price):
    """Calculate key financial ratios"""
    ratios = {}

    try:
        if balance_sheet is not None and not balance_sheet.empty:
            # Liquidity ratios
            current_assets = balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else 0
            current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else 0
            inventory = balance_sheet.loc['Inventory'].iloc[0] if 'Inventory' in balance_sheet.index else 0

            if current_liabilities > 0:
                ratios['Current Ratio'] = current_assets / current_liabilities
                ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities

            # Solvency ratios
            total_debt = balance_sheet.loc['Total Liab'].iloc[0] if 'Total Liab' in balance_sheet.index else 0
            total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0

            if total_equity > 0:
                ratios['Debt to Equity'] = total_debt / total_equity
            if total_assets > 0:
                ratios['Debt to Assets'] = total_debt / total_assets

        if income_stmt is not None and not income_stmt.empty:
            # Profitability ratios
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            total_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0

            if total_revenue > 0:
                ratios['Profit Margin'] = net_income / total_revenue

            if balance_sheet is not None and not balance_sheet.empty:
                total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0

                if total_assets > 0:
                    ratios['ROA'] = net_income / total_assets
                if total_equity > 0:
                    ratios['ROE'] = net_income / total_equity

        # Valuation ratios
        if balance_sheet is not None and not balance_sheet.empty and income_stmt is not None and not income_stmt.empty:
            shares_outstanding = balance_sheet.loc['Common Stock'].iloc[0] if 'Common Stock' in balance_sheet.index else 0
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0

            if shares_outstanding > 0 and current_price > 0:
                eps = net_income / shares_outstanding
                ratios['P/E'] = current_price / eps if eps > 0 else 0

                bv_per_share = total_equity / shares_outstanding if total_equity > 0 else 0
                ratios['P/B'] = current_price / bv_per_share if bv_per_share > 0 else 0

    except Exception as e:
        st.warning(f"Error calculating some ratios: {str(e)}")

    return ratios

# Forecasting function
@st.cache_data
def forecast_revenue(income_stmt, periods=5):
    """Forecast future revenue using Prophet"""
    try:
        if 'Total Revenue' not in income_stmt.index:
            return None

        revenue_series = income_stmt.loc['Total Revenue']
        dates = pd.to_datetime(revenue_series.index)
        df = pd.DataFrame({'ds': dates, 'y': revenue_series.values})
        df = df.sort_values('ds')

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq='Y')
        forecast = model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.warning(f"Error forecasting revenue: {str(e)}")
        return None

# DCF Valuation
def calculate_intrinsic_value(forecasted_fcf, discount_rate=0.1, terminal_growth=0.03, shares_outstanding=1e9):
    """Calculate intrinsic value using DCF"""
    try:
        if forecasted_fcf is None or forecasted_fcf.empty:
            return None

        fcf_values = forecasted_fcf['yhat'].values
        dcf = 0

        for i, fcf in enumerate(fcf_values):
            dcf += fcf / ((1 + discount_rate) ** (i + 1))

        terminal_value = fcf_values[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        dcf += terminal_value / ((1 + discount_rate) ** len(fcf_values))

        return dcf / shares_outstanding
    except Exception as e:
        st.warning(f"Error calculating intrinsic value: {str(e)}")
        return None

# Technical analysis
def calculate_technical_indicators(hist_data):
    """Calculate technical indicators"""
    indicators = {}

    try:
        if hist_data is not None and not hist_data.empty:
            # RSI
            indicators['RSI'] = ta.momentum.RSIIndicator(hist_data['Close']).rsi()

            # MACD
            macd = ta.trend.MACD(hist_data['Close'])
            indicators['MACD'] = macd.macd()
            indicators['MACD_Signal'] = macd.macd_signal()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(hist_data['Close'])
            indicators['BB_Upper'] = bb.bollinger_hband()
            indicators['BB_Lower'] = bb.bollinger_lband()
            indicators['BB_Middle'] = bb.bollinger_mavg()
    except Exception as e:
        st.warning(f"Error calculating technical indicators: {str(e)}")

    return indicators

# Main app
def main():
    # Header with right corner elements
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown('<div class="main-header">Shikher\'s Financial 2-Analysis App</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive Financial Analysis and Forecasting</div>', unsafe_allow_html=True)

    with col2:
        # Right corner elements
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("👤", help="Profile"):
                st.info("Profile feature coming soon!")
        with col_b:
            if st.button("🔔", help="Notifications"):
                st.info("Notifications feature coming soon!")
        with col_c:
            if st.button("🌙", help="Theme Toggle"):
                st.info("Dark/Light theme toggle coming soon!")

    # Sidebar Navigation
    st.sidebar.markdown("## 🧭 Navigation")

    # Navigation menu
    selected_page = st.sidebar.radio(
        "Select Section:",
        ["🏠 Dashboard", "📊 Financial Statements", "📈 Ratios & Metrics", "🔮 Forecasting", "💰 Valuation", "📉 Technical Analysis", "📋 Research Report", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    # Input section
    st.sidebar.subheader("Stock Selection")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS)", value="AAPL").upper()

    # Year selection
    st.sidebar.subheader("Analysis Parameters")
    years = st.sidebar.slider("Financial Years to Analyze", min_value=1, max_value=10, value=5, help="Number of recent financial years to include in analysis")

    if st.sidebar.button("🔍 Analyze Stock", type="primary"):
        with st.spinner("Fetching data and performing analysis..."):
            # Fetch data
            balance_sheet, income_stmt, cash_flow, hist_data, current_price, market, currency = fetch_financial_data(ticker, years)

            if balance_sheet is None or income_stmt is None:
                st.error("Unable to fetch financial data. Please check the ticker symbol and try again.")
                return

            # Calculate ratios
            ratios = calculate_ratios(balance_sheet, income_stmt, current_price)

            # Forecast revenue
            revenue_forecast = forecast_revenue(income_stmt)

            # Calculate intrinsic value (simplified)
            intrinsic_value = None
            if cash_flow is not None and not cash_flow.empty:
                try:
                    fcf_series = cash_flow.loc['Free Cash Flow'] if 'Free Cash Flow' in cash_flow.index else cash_flow.loc['Operating Cash Flow']
                    if not fcf_series.empty:
                        # Simple DCF calculation
                        latest_fcf = fcf_series.iloc[0]
                        growth_rate = 0.05  # Assumed growth rate
                        discount_rate = 0.1
                        terminal_value = latest_fcf * (1 + growth_rate) / (discount_rate - growth_rate)
                        dcf = sum([latest_fcf * (1 + growth_rate)**i / (1 + discount_rate)**i for i in range(1, 6)])
                        dcf += terminal_value / (1 + discount_rate)**5

                        # Estimate shares outstanding
                        market_cap = current_price * 1e9  # Rough estimate
                        shares_outstanding = market_cap / current_price
                        intrinsic_value = dcf / shares_outstanding if shares_outstanding > 0 else None
                except Exception as e:
                    st.warning(f"Error calculating intrinsic value: {str(e)}")

            # Technical analysis
            tech_indicators = calculate_technical_indicators(hist_data)

            # Store data in session state
            st.session_state.ticker = ticker
            st.session_state.current_price = current_price
            st.session_state.balance_sheet = balance_sheet
            st.session_state.income_stmt = income_stmt
            st.session_state.cash_flow = cash_flow
            st.session_state.hist_data = hist_data
            st.session_state.ratios = ratios
            st.session_state.revenue_forecast = revenue_forecast
            st.session_state.intrinsic_value = intrinsic_value
            st.session_state.tech_indicators = tech_indicators
            st.session_state.market = market
            st.session_state.currency = currency
            st.session_state.years = years

    # Display results if data is available
    if 'ticker' in st.session_state:
        display_results()

def display_results():
    """Display analysis results"""
    ticker = st.session_state.ticker
    current_price = st.session_state.current_price
    ratios = st.session_state.ratios
    intrinsic_value = st.session_state.intrinsic_value
    revenue_forecast = st.session_state.revenue_forecast
    tech_indicators = st.session_state.tech_indicators

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${current_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        market_cap = current_price * 1e9  # Rough estimate
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Market Cap (Est.)</div>
            <div class="metric-value">${market_cap/1e9:.1f}B</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if intrinsic_value:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Intrinsic Value</div>
                <div class="metric-value">${intrinsic_value:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Intrinsic Value</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if intrinsic_value and current_price:
            ratio = current_price / intrinsic_value
            if ratio < 0.9:
                rec_class = "recommendation-buy"
                rec_text = "BUY"
            elif ratio > 1.1:
                rec_class = "recommendation-sell"
                rec_text = "SELL"
            else:
                rec_class = "recommendation-hold"
                rec_text = "HOLD"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recommendation</div>
                <div class="metric-value {rec_class}">{rec_text}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Recommendation</div>
                <div class="metric-value">HOLD</div>
            </div>
            """, unsafe_allow_html=True)

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Financial Statements", "📈 Ratios & Metrics", "🔮 Forecasting", "💰 Valuation", "📉 Technical Analysis"])

    with tab1:
        st.subheader("Financial Statements")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Balance Sheet")
            if 'balance_sheet' in st.session_state and st.session_state.balance_sheet is not None:
                st.dataframe(st.session_state.balance_sheet.head(), use_container_width=True)
            else:
                st.write("No data available")

        with col2:
            st.subheader("Income Statement")
            if 'income_stmt' in st.session_state and st.session_state.income_stmt is not None:
                st.dataframe(st.session_state.income_stmt.head(), use_container_width=True)
            else:
                st.write("No data available")

        with col3:
            st.subheader("Cash Flow")
            if 'cash_flow' in st.session_state and st.session_state.cash_flow is not None:
                st.dataframe(st.session_state.cash_flow.head(), use_container_width=True)
            else:
                st.write("No data available")

    with tab2:
        st.subheader("Financial Ratios")

        if ratios:
            # Group ratios by category
            liquidity_ratios = {k: v for k, v in ratios.items() if k in ['Current Ratio', 'Quick Ratio']}
            profitability_ratios = {k: v for k, v in ratios.items() if k in ['ROA', 'ROE', 'Profit Margin']}
            solvency_ratios = {k: v for k, v in ratios.items() if k in ['Debt to Equity', 'Debt to Assets']}
            valuation_ratios = {k: v for k, v in ratios.items() if k in ['P/E', 'P/B']}

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Liquidity & Profitability")
                for category, ratio_dict in [("Liquidity", liquidity_ratios), ("Profitability", profitability_ratios)]:
                    st.write(f"**{category} Ratios:**")
                    for ratio_name, ratio_value in ratio_dict.items():
                        st.write(f"{ratio_name}: {ratio_value:.2f}")
                    st.write("")

            with col2:
                st.subheader("Solvency & Valuation")
                for category, ratio_dict in [("Solvency", solvency_ratios), ("Valuation", valuation_ratios)]:
                    st.write(f"**{category} Ratios:**")
                    for ratio_name, ratio_value in ratio_dict.items():
                        st.write(f"{ratio_name}: {ratio_value:.2f}")
                    st.write("")
        else:
            st.write("No ratio data available")

    with tab3:
        st.subheader("Revenue Forecasting")

        if revenue_forecast is not None:
            # Plot forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=revenue_forecast['ds'], y=revenue_forecast['yhat'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=revenue_forecast['ds'], y=revenue_forecast['yhat_upper'], fill=None, mode='lines', line_color='lightblue', name='Upper Bound'))
            fig.add_trace(go.Scatter(x=revenue_forecast['ds'], y=revenue_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='lightblue', name='Lower Bound'))

            fig.update_layout(
                title=f"{ticker} Revenue Forecast",
                xaxis_title="Date",
                yaxis_title="Revenue ($)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display forecast table
            st.subheader("Forecast Data")
            forecast_table = revenue_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
            forecast_table.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            st.dataframe(forecast_table, use_container_width=True)
        else:
            st.write("Unable to generate revenue forecast")

    with tab4:
        st.subheader("DCF Valuation")

        if intrinsic_value:
            st.success(f"**Intrinsic Value:** ${intrinsic_value:.2f}")
            st.info(f"**Current Price:** ${current_price:.2f}")

            if current_price > 0:
                upside = ((intrinsic_value - current_price) / current_price) * 100
                if upside > 0:
                    st.success(f"**Upside Potential:** {upside:.1f}%")
                else:
                    st.error(f"**Downside Risk:** {abs(upside):.1f}%")

            # Simple DCF explanation
            st.subheader("DCF Methodology")
            st.write("""
            **Discounted Cash Flow (DCF) Analysis:**
            - Projects future free cash flows
            - Discounts them to present value using WACC (10%)
            - Adds terminal value assuming perpetual growth (3%)
            - Divides by shares outstanding to get per-share value
            """)
        else:
            st.warning("Unable to calculate intrinsic value due to insufficient data")

    with tab5:
        st.subheader("Technical Analysis")

        if tech_indicators and 'hist_data' in st.session_state:
            hist_data = st.session_state.hist_data

            # Price chart with indicators
            fig = go.Figure()

            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name='Price'
            ))

            # Bollinger Bands
            if 'BB_Upper' in tech_indicators and 'BB_Lower' in tech_indicators:
                fig.add_trace(go.Scatter(x=hist_data.index, y=tech_indicators['BB_Upper'], line=dict(color='gray', width=1), name='BB Upper'))
                fig.add_trace(go.Scatter(x=hist_data.index, y=tech_indicators['BB_Lower'], line=dict(color='gray', width=1), name='BB Lower'))

            fig.update_layout(
                title=f"{ticker} Price Chart with Bollinger Bands",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # RSI Chart
            if 'RSI' in tech_indicators:
                st.subheader("RSI Indicator")
                fig_rsi = px.line(x=hist_data.index, y=tech_indicators['RSI'], title="Relative Strength Index (RSI)")
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)

                # RSI Signal
                latest_rsi = tech_indicators['RSI'].iloc[-1]
                if latest_rsi > 70:
                    st.error("⚠️ RSI indicates overbought conditions")
                elif latest_rsi < 30:
                    st.success("✅ RSI indicates oversold conditions")
                else:
                    st.info("ℹ️ RSI is neutral")

            # MACD Chart
            if 'MACD' in tech_indicators and 'MACD_Signal' in tech_indicators:
                st.subheader("MACD Indicator")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=hist_data.index, y=tech_indicators['MACD'], mode='lines', name='MACD'))
                fig_macd.add_trace(go.Scatter(x=hist_data.index, y=tech_indicators['MACD_Signal'], mode='lines', name='Signal'))
                fig_macd.update_layout(title="MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)

                # MACD Signal
                latest_macd = tech_indicators['MACD'].iloc[-1]
                latest_signal = tech_indicators['MACD_Signal'].iloc[-1]
                if latest_macd > latest_signal:
                    st.success("✅ MACD shows bullish momentum")
                else:
                    st.error("⚠️ MACD shows bearish momentum")
        else:
            st.write("Technical analysis data not available")

if __name__ == "__main__":
    main()
