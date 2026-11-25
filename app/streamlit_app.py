"""Small Streamlit client for manually testing the CivicAI FastAPI endpoints.

Run from the project root with:
    python -m streamlit run app\streamlit_app.py
"""

import json

import requests
import streamlit as st

st.set_page_config(page_title="CivicAI API tester")
st.title("CivicAI API tester")
st.caption("Send requests to the FastAPI service and inspect its responses.")

api_url = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000").strip().rstrip("/")
if not api_url:
    st.sidebar.error("Enter the base URL of the API.")
    st.stop()

health_col, info_col = st.columns(2)
if health_col.button("Check health"):
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        health_col.write(f"HTTP {response.status_code}")
        try:
            health_col.json(response.json())
        except ValueError:
            health_col.code(response.text or "(empty response)")
    except requests.RequestException as error:
        health_col.error(f"Could not reach the API: {error}")

if info_col.button("Get model info"):
    try:
        response = requests.get(f"{api_url}/model-info", timeout=10)
        info_col.write(f"HTTP {response.status_code}")
        try:
            info_col.json(response.json())
        except ValueError:
            info_col.code(response.text or "(empty response)")
    except requests.RequestException as error:
        info_col.error(f"Could not reach the API: {error}")

default_request = {
    "complaint_type": "Noise - Residential",
    "descriptor": "Loud Music/Party",
    "borough": "BROOKLYN",
    "location_type": "Residential Building/House",
    "open_data_channel_type": "PHONE",
    "incident_zip": "11201",
    "latitude": 40.69,
    "longitude": -73.99,
    "created_at": "2026-11-09T22:15:00",
}

request_text = st.text_area(
    "Request JSON",
    value=json.dumps(default_request, indent=2),
    height=320,
    help="Edit this JSON to try valid inputs or check how the API handles validation errors.",
)

if st.button("Send POST /predict", type="primary"):
    try:
        body = json.loads(request_text)
    except json.JSONDecodeError as error:
        st.error(f"Invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}")
    else:
        try:
            response = requests.post(f"{api_url}/predict", json=body, timeout=30)
            st.subheader(f"Response: HTTP {response.status_code}")
            try:
                st.json(response.json())
            except ValueError:
                st.code(response.text or "(empty response)")
        except requests.RequestException as error:
            st.error(f"Could not reach the API: {error}")
