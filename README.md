
# 🏥 Hospital Recommendation System (Sample Project)

This is a **ready-to-run website** that recommends hospitals based on your needs
(specialization, city, distance, rating, insurance, 24x7, cost).

![image alt](https://github.com/Mvsivateja/hospital_recommendation_system/blob/4b777a68a12fe01ca729b3f0dfbc4677717fd24d/1.png)

⚠️ **Note:** The dataset provided is **sample/fictional** for the towns you mentioned
(**Kadapa, Proddatur, Mydukur, Badvel, Jammalamadugu**). Replace `data/hospitals.csv`
with your real list to use in production.

## What you get
- **Streamlit app** — quickest way to get a website up.
- **Flask app** — classic website stack with templates.
- **Gradio app** — easiest to run/share from Google Colab.
- A reusable **recommender engine** (`recommender.py`).

---

## 1) Setup (Windows / macOS / Linux)

```bash
# Open a terminal in this folder
# 1) Create & activate a virtual environment (recommended)
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements.txt
```

## 2) Run the website (choose one)

### A) Streamlit (fastest)
```bash
streamlit run app_streamlit.py
```
Then open the local URL (usually http://localhost:8501).

### B) Flask (classic)
```bash
python app_flask.py
```
Then open http://127.0.0.1:5000

### C) Gradio (best for Google Colab)
```bash
python app_gradio.py
```
It will print a public share link in the terminal/Colab cell.

> If using **IDLE**: you can right-click `app_flask.py` → Run, or run `app_streamlit.py`
from a terminal. For **Colab**, upload all files, then run `!pip install -r requirements.txt`
and `!python app_gradio.py` in a cell.

---

## 3) How the recommender works (simple but effective)

We build a **text description** per hospital by combining *specializations + services + name + city*,
vectorize it with **TF–IDF**, and compare it with your **query** (selected specializations + optional text).
We also factor in:

- **Rating** (higher is better)
- **Distance** (closer is better, exponential decay)
- **Cost** (lower fee is better)
- **24x7 / Insurance** (bonus if required)

All of these combine into a final **score**, and we return the **Top N** results.

You can modify the weights inside `recommender.py`.

---

## 4) Replace with your real data

Open `data/hospitals.csv` and replace the rows with your actual hospitals.
**Keep the headers the same.** At minimum, fill: `name,city,specializations,services,rating,avg_fee,is_24x7,accepts_insurance,distance_km`.

---

## 5) Deploy options

- **Streamlit Community Cloud** — deploy `app_streamlit.py` directly from your GitHub repo.
- **Render** (Flask) — uses `Procfile` + `gunicorn app_flask:app`.
- **Hugging Face Spaces** — Streamlit or Gradio app.

For production, make sure your CSV is accurate and kept up to date.

---

## Project tree
```
hospital-reco-site/
├─ data/
│  └─ hospitals.csv             # sample/fictional data, replace with real
├─ templates/
│  └─ index.html                # Flask UI
├─ recommender.py               # core ranking logic
├─ app_streamlit.py             # Streamlit site
├─ app_flask.py                 # Flask site
├─ app_gradio.py                # Gradio UI (great for Colab)
├─ requirements.txt
├─ Procfile                     # for Render (Flask)
└─ README.md
```
