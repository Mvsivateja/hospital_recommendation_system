
import streamlit as st
import pandas as pd
from recommender import load_hospitals, build_index, recommend

st.set_page_config(page_title="Hospital Recommendation", page_icon="🏥", layout="wide")

st.title("🏥 Hospital Recommendation System")
st.caption("Sample dataset for Kadapa, Proddatur, Mydukur, Badvel, Jammalamadugu — replace data/hospitals.csv with your real data.")

df = load_hospitals()
vec, X = build_index(df)

cities = ["Any"] + sorted(df["city"].dropna().unique().tolist())
specs_all = sorted(set(sum([s.split("|") for s in df["specializations"].fillna("").tolist()], [])))

with st.sidebar:
    st.header("Your needs")
    pick_cities = st.multiselect("Preferred cities", cities, default=["Any"])
    pick_specs = st.multiselect("Specializations", specs_all, default=[])
    query_text = st.text_input("Optional symptoms / keywords", placeholder="e.g., chest pain, fracture, diabetes")
    min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1)
    max_fee = st.number_input("Max consultation fee (₹)", min_value=0, value=1000, step=50)
    max_dist = st.slider("Max distance (km)", 0.0, 50.0, 20.0, 0.5)
    need_24x7 = st.checkbox("Require 24x7", value=False)
    need_ins = st.checkbox("Require Insurance Acceptance", value=False)
    top_k = st.slider("Number of results", 1, 20, 10)

btn = st.button("Find hospitals")

if btn:
    results = recommend(
        df, vec, X,
        selected_specs=pick_specs,
        query_text=query_text,
        cities=pick_cities,
        min_rating=min_rating,
        max_fee=max_fee,
        need_24x7=need_24x7,
        need_insurance=need_ins,
        max_distance_km=max_dist,
        top_k=top_k,
    )
    st.success(f"Found {len(results)} result(s).")
    st.dataframe(results.reset_index(drop=True))
else:
    st.info("Set your preferences in the left panel, then click **Find hospitals**.")
