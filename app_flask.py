
from flask import Flask, render_template, request
import pandas as pd
from recommender import load_hospitals, build_index, recommend

app = Flask(__name__)
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

@app.route("/", methods=["GET","POST"])
def index():
    cities = ["Any"] + sorted(df["city"].dropna().unique().tolist())
    specs_all = get_specs_list(df)
    results = None
    form = {
        "cities": [], "specs": [], "query_text": "", "min_rating": "0",
        "max_fee": "1000", "max_dist": "20", "need_24x7": False, "need_ins": False, "top_k": "10"
    }

    if request.method == "POST":
        form["cities"] = request.form.getlist("cities")
        form["specs"] = request.form.getlist("specs")
        form["query_text"] = request.form.get("query_text","")
        form["min_rating"] = request.form.get("min_rating","0")
        form["max_fee"] = request.form.get("max_fee","1000")
        form["max_dist"] = request.form.get("max_dist","20")
        form["need_24x7"] = request.form.get("need_24x7") == "on"
        form["need_ins"] = request.form.get("need_ins") == "on"
        form["top_k"] = request.form.get("top_k","10")

        results = recommend(
            df, vec, X,
            selected_specs=form["specs"],
            query_text=form["query_text"],
            cities=form["cities"],
            min_rating=float(form["min_rating"] or 0),
            max_fee=float(form["max_fee"] or 1e9),
            need_24x7=form["need_24x7"],
            need_insurance=form["need_ins"],
            max_distance_km=float(form["max_dist"] or 1e9),
            top_k=int(form["top_k"] or 10),
        )

    return render_template("index.html", cities=cities, specs_all=specs_all, results=None if results is None else results.to_dict(orient="records"), form=form)

if __name__ == "__main__":
    app.run(debug=True)
