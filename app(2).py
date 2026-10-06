import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="A Marketing | AI Marketing Analyst",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #dff6ff 0%,
        #fce4ec 50%,
        #ffffff 100%
    );
}

/* Main header */
.main-header {
    background: linear-gradient(
        135deg,
        #bdefff,
        #ffd6e7
    );
    padding: 28px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.10);
}

.main-title {
    color: #000000;
    font-size: 42px;
    font-weight: 800;
    margin: 0;
}

.sub-title {
    color: #222222;
    font-size: 17px;
    margin-top: 8px;
}

/* Cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #eeeeee;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #555555;
    font-weight: 600;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #000000;
    margin-top: 6px;
}

/* Sections */
.section-box {
    background: rgba(255,255,255,0.92);
    padding: 22px;
    border-radius: 20px;
    margin-top: 18px;
    margin-bottom: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.07);
}

.section-title {
    color: #000000;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        #9ee8ff,
        #ffc1d9
    );
    color: #000000;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
}

.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #bdefff,
        #ffd6e7
    );
    color: #000000;
    font-weight: 700;
    border: none;
    border-radius: 12px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 18px;
    padding: 15px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: white;
    border-radius: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #dff6ff,
        #ffe6ef
    );
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <div class="main-title">💙 A Marketing</div>
    <div class="sub-title">
        AI Marketing Analyst — Upload your marketing data and
        automatically clean, analyze and understand your campaign performance.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column_name(col):
    """Make column names clean and consistent."""
    col = str(col).strip()
    col = col.lower()
    col = col.replace("-", "_")
    col = col.replace("/", "_")
    col = col.replace(" ", "_")
    col = "".join(c for c in col if c.isalnum() or c == "_")
    return col


def clean_dataframe(df):

    df = df.copy()

    # Remove completely empty rows/columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Clean column names
    original_columns = list(df.columns)

    new_columns = []
    for c in df.columns:
        new_columns.append(normalize_column_name(c))

    df.columns = new_columns

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Strip strings
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Convert common numeric columns
    numeric_keywords = [
        "revenue", "sales", "income", "amount", "cost",
        "spend", "budget", "profit", "click", "impression",
        "conversion", "conversions", "leads", "orders",
        "customers", "reach", "views", "ctr", "roi"
    ]

    for col in df.columns:
        col_lower = col.lower()

        if any(k in col_lower for k in numeric_keywords):
            if df[col].dtype == "object":

                cleaned = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace("₹", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )

                numeric_version = pd.to_numeric(
                    cleaned,
                    errors="coerce"
                )

                valid_ratio = numeric_version.notna().mean()

                if valid_ratio >= 0.50:
                    df[col] = numeric_version

    # Try converting date-like columns
    date_keywords = [
        "date", "day", "month", "year",
        "time", "timestamp", "created", "start", "end"
    ]

    for col in df.columns:
        col_lower = col.lower()

        if any(k in col_lower for k in date_keywords):

            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            if converted.notna().mean() >= 0.50:
                df[col] = converted

    # Fill numeric missing values with median
    for col in df.select_dtypes(include=np.number).columns:

        if df[col].isna().any():
            median_value = df[col].median()

            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)
            else:
                df[col] = df[col].fillna(0)

    # Fill text missing values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(
            ["nan", "None", "none", ""],
            np.nan
        )

        df[col] = df[col].fillna("Unknown")

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df


def find_column(df, keywords):

    columns = list(df.columns)

    # Exact keyword match
    for keyword in keywords:
        for col in columns:
            if col.lower() == keyword.lower():
                return col

    # Partial match
    for keyword in keywords:
        for col in columns:
            if keyword.lower() in col.lower():
                return col

    return None


def money(value):

    try:
        return f"₹{value:,.2f}"
    except:
        return str(value)


def number(value):

    try:
        return f"{value:,.0f}"
    except:
        return str(value)


def percentage(value):

    try:
        return f"{value:.2f}%"
    except:
        return str(value)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💙 A Marketing")

    st.markdown("""
    **AI Marketing Analyst**

    Upload your unclean marketing CSV.

    The system automatically:

    ✓ Cleans the data
    ✓ Detects important columns
    ✓ Analyzes revenue
    ✓ Analyzes channels
    ✓ Analyzes conversions
    ✓ Finds trends
    ✓ Creates charts
    ✓ Generates insights
    """)

    st.divider()

    st.markdown("### 📂 Upload Data")

    uploaded_file = st.file_uploader(
        "Upload Marketing CSV",
        type=["csv"]
    )


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.markdown("""
    <div class="section-box">

    <div class="section-title">
    🚀 Welcome to A Marketing
    </div>

    <p style="font-size:17px;">
    Upload your marketing CSV from the left sidebar.
    You do NOT need to manually clean the data first.
    </p>

    <p>
    The bot will automatically process your uploaded file
    and create a marketing performance dashboard.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# LOAD FILE
# ============================================================

try:

    df_original = pd.read_csv(uploaded_file)

except Exception:

    try:
        uploaded_file.seek(0)
        df_original = pd.read_csv(
            uploaded_file,
            encoding="latin1"
        )

    except Exception as e:

        st.error(
            "Unable to read this CSV file. "
            "Please check that the file is a valid CSV."
        )

        st.exception(e)
        st.stop()


# ============================================================
# AUTOMATIC CLEANING
# ============================================================

with st.spinner("🤖 A Marketing AI is cleaning and analyzing your data..."):

    df = clean_dataframe(df_original)


# ============================================================
# DETECT COLUMNS
# ============================================================

channel_col = find_column(
    df,
    [
        "channel",
        "marketing_channel",
        "media_channel",
        "platform",
        "source",
        "medium"
    ]
)

revenue_col = find_column(
    df,
    [
        "revenue",
        "sales",
        "total_sales",
        "income",
        "amount",
        "revenue_generated",
        "sales_revenue"
    ]
)

conversion_col = find_column(
    df,
    [
        "conversion",
        "conversions",
        "converted",
        "conversion_count",
        "orders",
        "purchases",
        "leads"
    ]
)

date_col = find_column(
    df,
    [
        "date",
        "campaign_date",
        "start_date",
        "day",
        "timestamp",
        "month"
    ]
)

campaign_col = find_column(
    df,
    [
        "campaign",
        "campaign_name",
        "campaign_id",
        "campaign_title"
    ]
)

cost_col = find_column(
    df,
    [
        "cost",
        "spend",
        "ad_spend",
        "marketing_spend",
        "budget"
    ])


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    "✅ File uploaded successfully — A Marketing automatically cleaned the data."
)


# ============================================================
# DETECTED DATA INFO
# ============================================================

st.markdown("""
<div class="section-box">
<div class="section-title">🔎 Automatically Detected Fields</div>
</div>
""", unsafe_allow_html=True)

detect_col1, detect_col2, detect_col3, detect_col4 = st.columns(4)

with detect_col1:
    st.metric(
        "Channel",
        channel_col if channel_col else "Not detected"
    )

with detect_col2:
    st.metric(
        "Revenue",
        revenue_col if revenue_col else "Not detected"
    )

with detect_col3:
    st.metric(
        "Conversion",
        conversion_col if conversion_col else "Not detected"
    )

with detect_col4:
    st.metric(
        "Date",
        date_col if date_col else "Not detected"
    )


# ============================================================
# TOP METRICS
# ============================================================

total_rows = len(df)

if revenue_col and pd.api.types.is_numeric_dtype(df[revenue_col]):
    total_revenue = df[revenue_col].sum()
    avg_revenue = df[revenue_col].mean()
else:
    total_revenue = 0
    avg_revenue = 0

if conversion_col and pd.api.types.is_numeric_dtype(df[conversion_col]):
    total_conversions = df[conversion_col].sum()
else:
    total_conversions = 0

if cost_col and pd.api.types.is_numeric_dtype(df[cost_col]):
    total_cost = df[cost_col].sum()
else:
    total_cost = 0


# ============================================================
# DASHBOARD CARDS
# ============================================================

st.markdown("""
<div class="section-box">
<div class="section-title">📊 Marketing Overview</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">TOTAL ROWS</div>
            <div class="metric-value">{number(total_rows)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">TOTAL REVENUE</div>
            <div class="metric-value">{money(total_revenue)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">CONVERSIONS</div>
            <div class="metric-value">{number(total_conversions)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">MARKETING COST</div>
            <div class="metric-value">{money(total_cost)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CLEANED DATA
# ============================================================

st.markdown("""
<div class="section-box">
<div class="section-title">🧹 Cleaned Marketing Data</div>
</div>
""", unsafe_allow_html=True)

st.dataframe(
    df.head(100),
    use_container_width=True
)


# ============================================================
# DOWNLOAD CLEANED CSV
# ============================================================

cleaned_csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Cleaned CSV",
    data=cleaned_csv,
    file_name="A_Marketing_Cleaned_Data.csv",
    mime="text/csv"
)


# ============================================================
# CHANNEL ANALYSIS
# ============================================================

if channel_col:

    st.markdown("""
    <div class="section-box">
    <div class="section-title">📱 Channel Performance Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    channel_summary = df.groupby(
        channel_col,
        dropna=False
    ).size().reset_index(name="records")

    if revenue_col and pd.api.types.is_numeric_dtype(df[revenue_col]):

        revenue_summary = df.groupby(
            channel_col,
            dropna=False
        )[revenue_col].sum().reset_index()

        revenue_summary = revenue_summary.sort_values(
            revenue_col,
            ascending=False
        )

        col_a, col_b = st.columns(2)

        with col_a:

            fig = px.bar(
                revenue_summary,
                x=channel_col,
                y=revenue_col,
                title="Revenue by Marketing Channel",
                text_auto=".2s"
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col_b:

            fig2 = px.pie(
                revenue_summary,
                names=channel_col,
                values=revenue_col,
                title="Revenue Share by Channel"
            )

            fig2.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        st.markdown("### Revenue by Channel")

        display_revenue = revenue_summary.copy()
        display_revenue["Revenue"] = display_revenue[
            revenue_col
        ].apply(money)

        st.dataframe(
            display_revenue[
                [channel_col, "Revenue"]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        fig = px.bar(
            channel_summary,
            x=channel_col,
            y="records",
            title="Records by Marketing Channel",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# CONVERSION ANALYSIS
# ============================================================

if conversion_col:

    st.markdown("""
    <div class="section-box">
    <div class="section-title">🎯 Conversion Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    if pd.api.types.is_numeric_dtype(
        df[conversion_col]
    ):

        total_conv = df[conversion_col].sum()

        if channel_col:

            conversion_summary = df.groupby(
                channel_col
            )[conversion_col].sum().reset_index()

            conversion_summary = conversion_summary.sort_values(
                conversion_col,
                ascending=False
            )

            fig = px.bar(
                conversion_summary,
                x=channel_col,
                y=conversion_col,
                title="Conversions by Channel",
                text_auto=".2s"
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.metric(
                "Total Conversions",
                number(total_conv)
            )


# ============================================================
# TREND ANALYSIS
# ============================================================

if date_col:

    st.markdown("""
    <div class="section-box">
    <div class="section-title">📈 Marketing Trend Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    try:

        trend_df = df.copy()

        trend_df[date_col] = pd.to_datetime(
            trend_df[date_col],
            errors="coerce"
        )

        trend_df = trend_df.dropna(
            subset=[date_col]
        )

        if len(trend_df) > 0:

            trend_df["Trend_Date"] = trend_df[
                date_col
            ].dt.date

            if revenue_col and pd.api.types.is_numeric_dtype(
                trend_df[revenue_col]
            ):

                trend_revenue = trend_df.groupby(
                    "Trend_Date"
                )[revenue_col].sum().reset_index()

                fig = px.line(
                    trend_revenue,
                    x="Trend_Date",
                    y=revenue_col,
                    markers=True,
                    title="Revenue Trend Over Time"
                )

                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            if conversion_col and pd.api.types.is_numeric_dtype(
                trend_df[conversion_col]
            ):

                trend_conversion = trend_df.groupby(
                    "Trend_Date"
                )[conversion_col].sum().reset_index()

                fig2 = px.line(
                    trend_conversion,
                    x="Trend_Date",
                    y=conversion_col,
                    markers=True,
                    title="Conversion Trend Over Time"
                )

                fig2.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

    except Exception:

        st.info(
            "Date column was detected, but a reliable trend could not be generated."
        )


# ============================================================
# AI INSIGHTS ENGINE
# ============================================================

st.markdown("""
<div class="section-box">
<div class="section-title">🤖 A Marketing AI Insights</div>
</div>
""", unsafe_allow_html=True)

insights = []
recommendations = []


# Revenue insights
if revenue_col and pd.api.types.is_numeric_dtype(df[revenue_col]):

    if channel_col:

        rev = df.groupby(
            channel_col
        )[revenue_col].sum().sort_values(
            ascending=False
        )

        if len(rev) > 0:

            top_channel = rev.index[0]
            top_value = rev.iloc[0]

            insights.append(
                f"💰 The channel generating the highest total revenue "
                f"is **{top_channel}**, with approximately "
                f"**{money(top_value)}**."
            )

            recommendations.append(
                f"Consider reviewing the campaigns under **{top_channel}** "
                f"to understand which activities are contributing to its revenue."
            )


# Conversion insights
if conversion_col and pd.api.types.is_numeric_dtype(
    df[conversion_col]
):

    if channel_col:

        conv = df.groupby(
            channel_col
        )[conversion_col].sum().sort_values(
            ascending=False
        )

        if len(conv) > 0:

            best_conversion_channel = conv.index[0]
            best_conversion_value = conv.iloc[0]

            insights.append(
                f"🎯 **{best_conversion_channel}** has the highest "
                f"recorded conversions, with **{number(best_conversion_value)}**."
            )

            recommendations.append(
                f"Analyze the messaging, audience and campaign structure "
                f"used by **{best_conversion_channel}** and compare it with other channels."
            )


# Cost / revenue insight
if (
    revenue_col
    and cost_col
    and pd.api.types.is_numeric_dtype(df[revenue_col])
    and pd.api.types.is_numeric_dtype(df[cost_col])
):

    if total_cost > 0:

        roi_value = (
            (total_revenue - total_cost)
            / total_cost
        ) * 100

        insights.append(
            f"📊 Based on the uploaded revenue and cost fields, "
            f"the calculated return relative to marketing cost is "
            f"approximately **{roi_value:.2f}%**."
        )

        if roi_value >= 0:

            recommendations.append(
                "Review which campaigns are contributing most to the "
                "positive return and use those findings when planning future campaigns."
            )

        else:

            recommendations.append(
                "Review high-cost campaigns and compare their revenue and "
                "conversion contribution before increasing their budget."
            )


# Dataset insight
insights.append(
    f"📁 The uploaded dataset contains **{number(total_rows)} records** "
    f"and **{number(len(df.columns))} columns** after automatic cleaning."
)


# Missing-value insight
missing_values = df.isna().sum().sum()

if missing_values == 0:

    insights.append(
        "🧹 The cleaned dataset currently contains no remaining missing values."
    )

else:

    insights.append(
        f"🧹 The cleaning process identified remaining missing values: "
        f"**{number(missing_values)}**."
    )


# Trend insight
if date_col and revenue_col:

    try:

        temp = df.copy()

        temp[date_col] = pd.to_datetime(
            temp[date_col],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=[date_col]
        )

        if len(temp) >= 2:

            temp["period"] = temp[date_col].dt.date

            trend = temp.groupby(
                "period"
            )[revenue_col].sum()

            if len(trend) >= 2:

                first_value = trend.iloc[0]
                last_value = trend.iloc[-1]

                if first_value != 0:

                    change = (
                        (last_value - first_value)
                        / abs(first_value)
                    ) * 100

                    direction = (
                        "increased"
                        if change >= 0
                        else "decreased"
                    )

                    insights.append(
                        f"📈 Revenue **{direction}** by approximately "
                        f"**{abs(change):.2f}%** between the first and last "
                        f"available periods in the uploaded data."
                    )

    except Exception:
        pass


# Display insights
for item in insights:

    st.markdown(
        f"""
        <div style="
            background:white;
            padding:15px;
            margin:8px 0;
            border-radius:14px;
            border-left:5px solid #9ee8ff;
            color:#000000;
        ">
        {item}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.markdown("### 💡 Recommendations")

if len(recommendations) == 0:

    recommendations.append(
        "Upload a dataset containing revenue, channel, conversion and date fields "
        "for deeper marketing recommendations."
    )

for recommendation in recommendations:

    st.markdown(
        f"""
        <div style="
            background:#fff5f9;
            padding:15px;
            margin:8px 0;
            border-radius:14px;
            border-left:5px solid #ffc1d9;
            color:#000000;
        ">
        {recommendation}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA QUALITY REPORT
# ============================================================

st.markdown("""
<div class="section-box">
<div class="section-title">🔍 Data Quality Report</div>
</div>
""", unsafe_allow_html=True)

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:

    st.metric(
        "Original Rows",
        len(df_original)
    )

with quality_col2:

    st.metric(
        "Cleaned Rows",
        len(df)
    )

with quality_col3:

    st.metric(
        "Columns",
        len(df.columns)
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

with st.expander("📋 View Detected Column Information"):

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "Missing Values": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Unique Values": [
            int(df[c].nunique())
            for c in df.columns
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<br>
<div style="
    text-align:center;
    padding:20px;
    color:#000000;
    font-weight:600;
">
    💙 A Marketing | AI Marketing Analyst
    <br>
    <span style="font-size:13px;">
    Automatic Cleaning • Revenue • Channels • Conversions • Trends
    </span>
</div>
""", unsafe_allow_html=True)
