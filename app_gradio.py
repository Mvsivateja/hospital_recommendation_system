
import gradio as gr
import pandas as pd
from recommender import load_hospitals, build_index, recommend

df = load_hospitals()
vec, X = build_index(df)

def get_specs_list(df):
    specs = set()
    for s in df["specializations"].fillna("").tolist():
        for x in s.split("|"):
            x = x.strip()
            if x:
                specs.add(x)
    return sorted(specs)

cities_list = ["Any"] + sorted(df["city"].dropna().unique().tolist())
specs_all = get_specs_list(df)

def infer(cities, specs, query_text, min_rating, max_fee, max_dist, need_24x7, need_ins, top_k):
    results = recommend(
        df, vec, X,
        selected_specs=specs,
        query_text=query_text,
        cities=cities,
        min_rating=min_rating,
        max_fee=max_fee,
        need_24x7=need_24x7,
        need_insurance=need_ins,
        max_distance_km=max_dist,
        top_k=int(top_k),
    )
    return results

with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Hospital Recommendation System")
    gr.Markdown("Sample data — replace `data/hospitals.csv` with your real list.")
    with gr.Row():
        cities = gr.CheckboxGroup(choices=cities_list, label="Preferred cities", value=["Any"])
        specs = gr.CheckboxGroup(choices=specs_all, label="Specializations")
    query_text = gr.Textbox(label="Optional symptoms/keywords", placeholder="e.g., chest pain, fracture, diabetes")
    with gr.Row():
        min_rating = gr.Slider(0, 5, value=0, step=0.1, label="Minimum rating")
        max_fee = gr.Number(value=1000, label="Max fee (₹)")
        max_dist = gr.Slider(0, 50, value=20, step=0.5, label="Max distance (km)")
    with gr.Row():
        need_24x7 = gr.Checkbox(label="Require 24x7")
        need_ins = gr.Checkbox(label="Require Insurance")
        top_k = gr.Slider(1, 20, value=10, step=1, label="Number of results")
    btn = gr.Button("Find hospitals")
    out = gr.Dataframe(interactive=False)
    btn.click(infer, [cities,specs,query_text,min_rating,max_fee,max_dist,need_24x7,need_ins,top_k], out)

if __name__ == "__main__":
    demo.launch()
