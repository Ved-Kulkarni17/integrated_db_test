import os
import streamlit as st
import json
import base64
import requests
import pandas as pd
import plotly.express as px
import logging
import mlflow
from typing import List
from dotenv import load_dotenv

load_dotenv()

from model_serving_utils import query_endpoint

# ───── CONFIG ─────
st.set_page_config(page_title="Databricks Assistant", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───── SIDEBAR ─────
st.sidebar.title("🧠 Mode Selector")
mode = st.sidebar.radio("Choose Mode", ["Unstructured", "Structured"])

if mode == "Unstructured":
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN_U")
    SERVING_ENDPOINT = "databricks_RAG"
else:
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN_S")
    SERVING_ENDPOINT = "databricks_structured"
    
mlflow.set_tracking_uri("databricks")

# ───── UNSTRUCTURED MODE ─────
if mode == "Unstructured":
    st.title("📄→🧠 RAG Chatbot")
    st.markdown("Upload a PDF → Ask questions → Get answers via your Databricks Serving Endpoint.")

    if "encoded_pdf" not in st.session_state:
        st.session_state["encoded_pdf"] = None
    if "file_name" not in st.session_state:
        st.session_state["file_name"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        st.session_state["encoded_pdf"] = base64.b64encode(pdf_bytes).decode()
        st.session_state["file_name"] = uploaded_file.name
        st.success(f"**{uploaded_file.name}** uploaded — ask away!")

    if not st.session_state["encoded_pdf"]:
        st.info("👆 Please upload a PDF first")
        st.stop()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def query_rag_endpoint(encoded_pdf, file_name, question, history):
        url = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT}/invocations"
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json",
        }
        data = {
            "dataframe_records": [
                {
                    "encoded_pdf": encoded_pdf,
                    "file_name": file_name,
                    "question": question,
                    "chat_history": json.dumps(history),
                }
            ]
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            st.error(f"❌ {resp.status_code} {resp.reason}")
            st.text(resp.text)
            raise
        prediction = resp.json()["predictions"][0]
        return next(iter(prediction.values())) if isinstance(prediction, dict) else str(prediction)

    if question := st.chat_input("Ask something about the PDF …"):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Answering …"):
                try:
                    answer = query_rag_endpoint(
                        st.session_state["encoded_pdf"],
                        st.session_state["file_name"],
                        question,
                        st.session_state["messages"]
                    )
                except Exception as e:
                    answer = f"❌ **Endpoint error:** {e}"
                st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    with st.sidebar.expander("🛠 Debug Info"):
        st.write("PDF:", st.session_state["file_name"])
        st.write("Turns:", len(st.session_state["messages"]))
        st.write("Endpoint:", SERVING_ENDPOINT)

# ───── STRUCTURED MODE ─────
else:
    st.title("📊 Structured SQL Assistant")
    st.markdown("Ask any question about the `sales2` table in natural language.")

    user_question = st.text_area("📝 Enter your question:", height=100, placeholder="e.g., What were the total sales for each product line in Q1?")
    generate_insights = st.checkbox("🧠 Generate insights from results", value=True)

    if st.button("Ask"):
        if not user_question.strip():
            st.error("❌ Please enter a question.")
        else:
            with st.spinner("🧮 Thinking..."):
                try:
                    response = query_endpoint(
                        endpoint_name=SERVING_ENDPOINT,
                        question=user_question,
                        generate_insights=generate_insights
                    )

                    st.subheader("🛠️ Generated SQL")
                    st.code(response.get("sql_query", ""), language="sql")

                    st.subheader("📄 Query Result")
                    st.text(response.get("query_result", "No results."))

                    if response.get("insights"):
                        st.subheader("🔍 Insights")
                        st.success(response["insights"])

                    vis = response.get("visualization_data", {})
                    chart_type = vis.get("chart_type", "none")
                    x_vals = vis.get("x")
                    y_vals = vis.get("y")
                    x_label = vis.get("x_label", "X")
                    y_label = vis.get("y_label", "Y")

                    st.subheader("📊 Visualization Suggestion")
                    if not x_vals or not y_vals:
                        st.info("ℹ️ No visualization available.")
                    else:
                        df = pd.DataFrame({x_label: x_vals, y_label: y_vals})
                        if chart_type == "bar":
                            fig = px.bar(df, x=x_label, y=y_label, text=y_label)
                        elif chart_type == "line":
                            fig = px.line(df, x=x_label, y=y_label, markers=True)
                        elif chart_type == "pie":
                            fig = px.pie(df, names=x_label, values=y_label)
                        elif chart_type == "trend":
                            df = pd.DataFrame({x_label: list(range(len(y_vals))), y_label: y_vals})
                            fig = px.line(df, x=x_label, y=y_label, markers=True)
                        elif chart_type == "distribution":
                            fig = px.bar(df, x=x_label, y=y_label, text=y_label)
                        else:
                            fig = None

                        if fig:
                            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("ℹ️ No visualization generated.")
                except Exception as e:
                    st.error("⚠️ Something went wrong.")
                    st.exception(e)
