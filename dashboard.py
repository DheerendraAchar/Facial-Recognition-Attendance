# dashboard.py

import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st.title("  Facial Attendance Dashboard")

# Load data
if os.path.exists("attendance.csv"):
    df = pd.read_csv("attendance.csv", header=None, names=["Student", "Timestamp", "Attention"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp", ascending=False)
else:
    st.warning("attendance.csv not found. Run the recognition system first.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.markdown("## Filters")
    students = df["Student"].unique()
    selected_student = st.selectbox("Filter by Student", options=["All"] + list(students))

    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()
    date_range = st.date_input("Date Range", [min_date, max_date])

    # Filter logic
    if selected_student != "All":
        df = df[df["Student"] == selected_student]
    df = df[(df["Timestamp"].dt.date >= date_range[0]) & (df["Timestamp"].dt.date <= date_range[1])]

st.markdown("###   Attendance Records")
st.dataframe(df, use_container_width=True)

# Attendance count summary
st.markdown("###   Attendance Frequency")
summary = df["Student"].value_counts().reset_index()
summary.columns = ["Student", "Sessions"]
st.bar_chart(summary.set_index("Student"))

# Live attention trend
st.markdown("###   Attention Trends Over Time")
if selected_student != "All" and not df.empty:
    trend = df.groupby(pd.Grouper(key="Timestamp", freq="1Min")).mean(numeric_only=True).reset_index()
    fig = px.line(trend, x="Timestamp", y="Attention", title=f"Attention Trend for {selected_student}",
                  labels={"Attention": "Avg Attention Score"})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select a specific student to view attention trends.")

# Export option
st.markdown("###   Export Filtered Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="attendance_filtered.csv", mime="text/csv")
