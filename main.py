# -*- coding: utf-8 -*-

# Core Imports
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# For AI/ML (NLP Question-Answering)
from transformers import pipeline

# -----------------------------
# 1. PAGE CONFIG & HEADER
# -----------------------------
st.set_page_config(
    page_title="Getting to Know Mutually Human",
    layout="wide",  # ensures wide layout
    page_icon="🤝"
)

# Retrieve Streamlit's theme settings to adapt plot backgrounds & fonts
theme_mode = st.get_option("theme.base")
if theme_mode == "dark":
    bg_color = "#0E1117"  # or "rgba(0, 0, 0, 0)" for transparency
    text_color = "white"
else:
    bg_color = "white"
    text_color = "black"

# Primary and secondary colors
primary_color = "#4B0082"   # dark purple
secondary_color = "#4B4B4B" # dark gray

# Header with Mutually Human branding
t1, t2 = st.columns((0.15, 0.85))
t1.image("index_0.png", width=120)  # Replace with actual logo path or URL
t2.title("Getting to Know Mutually Human")
t2.markdown(
    """
    This dashboard demonstrates my skill sets with Data Engineering, Visualization, and AI/ML components focusing on a case study regarding
    **Mutually Human**, a custom software consultancy headquartered in Grand Rapids, Michigan.
    """
)

# -----------------------------
# 2. COMPANY OVERVIEW METRICS
# -----------------------------
m1, m2, m3, m4, _ = st.columns((1, 1, 1, 1, 1))
m1.metric(label="Founded", value="2006")
m2.metric(label="Headquarters", value="Grand Rapids, MI")
m3.metric(label="Team Size (approx.)", value="50+")
m4.metric(label="Key Expertise", value="Human-Centered Design")

# -----------------------------
# 3. PREPARE HYPOTHETICAL MANUFACTURING DATA
# -----------------------------
manufacturing_csv = "manufacturing_data.csv"
if not os.path.exists(manufacturing_csv):
    # Create a fictional manufacturing dataset for demonstration
    data_mfg = {
        "Month": [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ],
        "ProductionVolume": [1200, 1350, 1100, 1450, 1500, 1600,
                             1550, 1580, 1490, 1700, 1650, 1800],
        "DefectRate": [0.02, 0.025, 0.03, 0.018, 0.025, 0.022,
                       0.028, 0.021, 0.027, 0.019, 0.03, 0.024],
        "DowntimeMinutes": [40, 50, 35, 60, 55, 45,
                            70, 65, 40, 55, 60, 50],
        "CostOfGoodsSold": [10000, 11000, 10500, 11500, 11800, 12000,
                            11900, 12100, 11750, 12500, 12300, 12800]
    }
    df_manufacturing = pd.DataFrame(data_mfg)
    df_manufacturing.to_csv(manufacturing_csv, index=False)
else:
    df_manufacturing = pd.read_csv(manufacturing_csv)

# -----------------------------
# 4. TABS FOR NAVIGATION
# -----------------------------
tab_overview, tab_manufacturing, tab_qa = st.tabs([
    "Overview",
    "Manufacturing Insights",
    "QA AI/ML Application"
])

# -----------------------------
# 4A. OVERVIEW TAB
# -----------------------------
with tab_overview:
    st.header("Application Overview")
    st.markdown(
        """
        **This Dashboard Demonstrates**:
        - **Data Engineering**: Ingesting or generating CSV data, 
          structuring and preparing it for visualization.
        - **AI/ML**: Simple context-based Q&A using the Hugging Face Transformers pipeline.
        - **Visualization**: Plotly and Streamlit integration for interactive charts.

        **About Mutually Human**:
        - Specializes in human-centered design and custom software development.
        - Known for deep collaboration and empathy-driven processes.
        - Helps organizations innovate with bespoke digital solutions.

        You can adapt or expand these components to create more complex 
        analytics dashboards, interactive ML applications, or advanced 
        data pipelines as needed.
        """
    )

# -----------------------------
# 4B. MANUFACTURING INSIGHTS TAB
# -----------------------------
with tab_manufacturing:
    st.header("Manufacturing Insights Dashboard")

    st.write(
        "A demonstration of medium-to-advanced BI & analytics capabilities for a "
        "hypothetical manufacturing scenario."
    )

    # 4B-1. KPI METRICS
    col1, col2, col3, col4 = st.columns(4)
    total_production = df_manufacturing["ProductionVolume"].sum()
    avg_defect = df_manufacturing["DefectRate"].mean() * 100
    total_downtime = df_manufacturing["DowntimeMinutes"].sum()
    total_cogs = df_manufacturing["CostOfGoodsSold"].sum()

    col1.metric("Total Production", f"{total_production:,}")
    col2.metric("Avg Defect Rate", f"{avg_defect:.2f}%")
    col3.metric("Total Downtime (min)", f"{total_downtime:,}")
    col4.metric("Total COGS", f"${total_cogs:,.2f}")

    # -------------------------
    # CHARTS LAYOUT (2 ROWS, 2 COLUMNS PER ROW)
    # -------------------------
    # ROW 1: Multi-axis line chart & Correlation heatmap
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Production Volume & Downtime Over Time")

        fig_mfg = go.Figure()

        # Production Volume trace (left axis)
        fig_mfg.add_trace(
            go.Scatter(
                x=df_manufacturing["Month"],
                y=df_manufacturing["ProductionVolume"],
                name="Production Volume",
                mode="lines+markers",
                line=dict(color=primary_color, width=2),
                marker=dict(size=6),
            )
        )

        # Downtime trace (right axis)
        fig_mfg.add_trace(
            go.Scatter(
                x=df_manufacturing["Month"],
                y=df_manufacturing["DowntimeMinutes"],
                name="Downtime (min)",
                mode="lines+markers",
                line=dict(color=primary_color, width=2, dash="dash"),
                marker=dict(size=6),
                yaxis="y2"
            )
        )

        fig_mfg.update_layout(
            title="Production vs. Downtime",
            xaxis=dict(
                title="Month",
                linecolor="black",
                linewidth=2,
                mirror=True
            ),
            yaxis=dict(
                title="Production Volume",
                linecolor="black",
                linewidth=2,
                mirror=True
            ),
            yaxis2=dict(
                title="Downtime Minutes",
                overlaying="y",
                side="right",
                linecolor="black",
                linewidth=2
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(family="Helvetica-Oblique", size=14, color=text_color),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig_mfg, use_container_width=True)

    with row1_col2:
        st.subheader("Correlation Heatmap")

        # Calculate correlation
        numeric_cols = ["ProductionVolume", "DefectRate", "DowntimeMinutes", "CostOfGoodsSold"]
        corr_matrix = df_manufacturing[numeric_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale=[secondary_color, primary_color],
            title="Correlation (Prod, Defects, Downtime, COGS)"
        )
        fig_corr.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(family="Helvetica-Oblique", size=14, color=text_color)
        )
        fig_corr.update_xaxes(linecolor="black", linewidth=2, mirror=True)
        fig_corr.update_yaxes(linecolor="black", linewidth=2, mirror=True)

        st.plotly_chart(fig_corr, use_container_width=True)

    # ROW 2: Sankey & Bar Chart
    row2_col1, row2_col2 = st.columns(2)

    # -------------------------
    # Visualization 3: Sankey Diagram
    # -------------------------
    with row2_col1:
        st.subheader("Mock Sankey Flow")
        # Define simple flow data (source -> target -> value)
        sankey_labels = ["Raw Materials", "In Production", "Assembly", "Finished Goods", "Defects"]
        sankey_source = [0, 1, 2, 2]  # indexes of sankey_labels
        sankey_target = [1, 2, 3, 4]
        sankey_values = [1000, 950, 900, 50]

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=2),
                label=sankey_labels,
                color=primary_color  # node color
            ),
            link=dict(
                source=sankey_source,
                target=sankey_target,
                value=sankey_values,
                color=secondary_color  # link color
            )
        )])

        fig_sankey.update_layout(
            title_text="Material Flow (Hypothetical)",
            font=dict(family="Helvetica-Oblique", size=14, color=text_color),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig_sankey, use_container_width=True)

    # -------------------------
    # Visualization 4: Bar Chart (Defect Rate by Month)
    # -------------------------
    with row2_col2:
        st.subheader("Defect Rate by Month")
        fig_bar = px.bar(
            df_manufacturing,
            x="Month",
            y="DefectRate",
            color_discrete_sequence=[primary_color],  # single color (dark purple)
            title="Defect Rate per Month"
        )
        fig_bar.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(family="Helvetica-Oblique", size=14, color=text_color),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig_bar.update_xaxes(
            title="Month",
            linecolor="black",
            linewidth=2,
            mirror=True
        )
        fig_bar.update_yaxes(
            title="Defect Rate",
            linecolor="black",
            linewidth=2,
            mirror=True,
            tickformat=".2%"  # convert decimal to percentage
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.write("---")
    with st.expander("View Raw Manufacturing Data", expanded=False):
        st.dataframe(df_manufacturing)

# -----------------------------
# 4C. QA AI/ML APPLICATION TAB
# -----------------------------
with tab_qa:
    st.header("Mutually Human QA System")
    st.write(
        "Ask any question about Mutually Human's projects, methodologies, or values, "
        "and get an AI-powered answer (demo)."
    )

    # QA Pipeline (using a pre-trained model)
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Context about Mutually Human (can be expanded or loaded from external source)
    mh_context = """
        Mutually Human is a custom software consultancy founded in 2006 
        in Grand Rapids, Michigan. The company specializes in human-centered design, 
        custom software development, and strategic consulting. 
        Mutually Human focuses on building intuitive, user-friendly solutions 
        across a variety of industries, partnering closely with clients 
        to create meaningful digital experiences. 
        Core values include collaboration, innovation, empathy, and craftsmanship.
    """

    # Text input for user question
    user_question = st.text_input(
        "Enter your question about Mutually Human:",
        placeholder="e.g., What is Mutually Human's approach to user experience?"
    )

    # Handle user question
    if user_question:
        # Pass the user question and context to the QA pipeline
        answer = qa_pipeline(
            question=user_question, 
            context=mh_context
        )
        st.write(f"**Answer:** {answer['answer']}")
    else:
        st.write("Enter a question above to receive an answer.")

    # Suggested questions
    st.write("### Suggested Questions:")
    st.write("- What is Mutually Human's approach to user experience design?")
    st.write("- How does Mutually Human help clients transform their digital processes?")
    st.write("- What are Mutually Human's core values?")
    st.write("- Which industries does Mutually Human serve?")
    st.write("- What differentiates Mutually Human from other consultancies?")
