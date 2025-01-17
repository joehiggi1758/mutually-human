# -*- coding: utf-8 -*-

# Core Imports
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# For AI/ML (NLP Question-Answering)
from transformers import pipeline

# -----------------------------
# 1. PAGE CONFIG & HEADER
# -----------------------------
st.set_page_config(
    page_title="Getting to Know Mutually Human",
    layout="wide",
    page_icon="🤝"
)

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
# 2. OVERVIEW METRICS
# -----------------------------
m1, m2, m3, m4, _ = st.columns((1, 1, 1, 1, 1))
m1.metric(label="Founded", value="2006")
m2.metric(label="Headquarters", value="Grand Rapids, MI")
m3.metric(label="Team Size (approx.)", value="50+")
m4.metric(label="Key Expertise", value="Human-Centered Design")

# -----------------------------
# 3. PREPARE DEMO DATA
# -----------------------------
# In a real scenario, replace with actual data or connect to a DB
demo_csv = "mutually_human_projects.csv"

# Only create CSV if it doesn't exist yet
if not os.path.exists(demo_csv):
    # Example data representing completed projects over years (fictional)
    data = {
        "Year": [2017, 2018, 2019, 2020, 2021, 2022, 2023],
        "Projects_Completed": [12, 15, 20, 22, 18, 25, 30]
    }
    df_projects = pd.DataFrame(data)
    df_projects.to_csv(demo_csv, index=False)
else:
    df_projects = pd.read_csv(demo_csv)

# -----------------------------
# 4. TABS FOR NAVIGATION
# -----------------------------
tab_qa, tab_projects, tab_overview = st.tabs([
    "QA AI/ML Application", 
    "Project Statistics",
    "Overview"
])

# -----------------------------
# 4A. QA AI/ML APPLICATION
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

# -----------------------------
# 4B. PROJECT STATISTICS TAB
# -----------------------------
with tab_projects:
    st.header("Project Statistics (Demo)")
    st.write(
        "Below is a sample of how we can display data engineering and visualization skills "
        "using fictional project data."
    )

    # Create a line chart to visualize projects completed over the years
    fig = px.line(
        df_projects,
        x="Year",
        y="Projects_Completed",
        title="Mutually Human - Projects Completed by Year (Fictional Data)",
        markers=True
    )

    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        font=dict(family="Helvetica", size=14, color="Black")
    )
    fig.update_traces(line=dict(width=3, color="#FFA500"), marker=dict(size=8))
    fig.update_yaxes(title="Projects Completed")
    fig.update_xaxes(title="Year")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Data Engineering** aspect: 
        - Demonstrating how a CSV can be generated or loaded, 
          then processed (e.g., merging, cleaning, or joining in real cases).

        **Visualization** aspect:
        - Using Plotly and Streamlit to present interactive charts.

        This setup can be extended to include metrics such as project durations, 
        budgets, or client satisfaction scores.
        """
    )

# -----------------------------
# 4C. OVERVIEW TAB
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
