# QeemaBank Loan Risk — FastAPI Deployment

A small FastAPI app that serves your two trained pipelines (default-risk
classifier + loan-amount regressor) behind a simple HTML form, ready to
deploy for free.

```
loan_app/
├── app/
│   ├── main.py              # FastAPI app (routes, prediction logic)
│   └── __init__.py
├── templates/
│   └── index.html           # the form + result page
├── static/
│   └── style.css
├── models/                  # <-- put your 2 exported .joblib files here
│   └── README.md
├── export_models_snippet.py # run this in Colab to create the .joblib files
├── requirements.txt
├── Dockerfile
└── README.md                # this file
```

Tested locally end-to-end (form load, form submit, JSON API, both the
"safe" and "risky" result states) before handing this to you.

---

## Step 0 — Export your trained models from Colab

Open `bank_loans.ipynb` in Colab. After the cells that create
`best_tuned_model` (cell 16, the tuned classifier) and `rf_reg_pipe`
(cell 14, the regressor) have run, add a **new cell** at the bottom with
the contents of `export_models_snippet.py` in this folder, and run it.

That will download two files to your computer:
- `classification_pipeline.joblib`
- `regression_pipeline.joblib`

**Note the scikit-learn version it prints.** If it differs from
`scikit-learn==1.5.1` in `requirements.txt`, update that line to match —
a version mismatch is the #1 reason a pickled model fails to load on a
different machine.

Move both files into the `models/` folder in this project.

---

## Step 1 — Run it locally first (2 minutes)

```bash
cd loan_app
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open **http://localhost:8000** — you should see the form. Submit it and
confirm you get a probability + suggested loan amount back. Also sanity
check the JSON API:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","age":35,"annual_income":60000,"employment_years":5,
       "education":"Bachelor","credit_score":680,"debt_to_income":0.3,
       "num_existing_loans":1,"loan_purpose":"Debt Consolidation"}'
```

**Before deploying**, open `app/main.py` and check
`LOAN_PURPOSE_OPTIONS` — I filled in placeholder categories since I
didn't have your actual CSV. Run `df['loan_purpose'].unique()` in your
notebook and swap in the real values (your `OneHotEncoder` has
`handle_unknown="ignore"`, so it won't crash on a mismatch, but the
dropdown should match what the model actually saw during training).

---

## Step 2 — Free deployment: use Render.com instead of Docker-based Hugging Face Spaces

As of now, Hugging Face Spaces that use **Docker** or **Gradio** are not free on the standard plan. For this project, the easiest free solution is to deploy the FastAPI app on Render instead of using a Docker Space.

This is the recommended free path for this repository.

1. Push this project (including the two `.joblib` files in `models/`)
   to a **GitHub repo**.
2. Go to **render.com** → sign up (GitHub login is easiest) → **New** →
   **Web Service**.
3. Connect your GitHub repo.
4. Configure:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance type**: Free
5. Click **Create Web Service**. Render builds and deploys
   automatically; you'll get a URL like
   `https://qeemabank-loan-risk.onrender.com`.
6. Every future `git push` auto-redeploys.

**Why this is the solution:**
- It avoids the paid Docker requirement on Hugging Face.
- It works with this FastAPI project directly.
- It is free for a small demo portfolio app.

**Note:**
- The free tier on Render sleeps after inactivity and cold-starts in ~30–60s.
- This is normal for free hosting and is fine for demos.

---

## Step 3 — If you specifically want a Hugging Face Space

A Hugging Face Space can still be used, but not as a **Docker/Gradio app** unless you pay. The free workaround is:

- create a **Static** Space on Hugging Face, and
- keep the actual model API hosted on Render (or another free Python host).

Then your Space can simply be a lightweight landing page or frontend that calls the Render URL.

This is a good choice if you want the project to appear under your Hugging Face profile without paying for Docker.

---

## Step 4 (optional) — Hugging Face + Render hybrid

1. Deploy the model backend on Render.
2. Create a Hugging Face **Static** Space.
3. Place a simple HTML page in that Space that links to your deployed API or embeds the frontend using JavaScript.
4. Your app stays free, and your Hugging Face profile still looks polished for a portfolio.

**If the build fails,** the usual causes are still:
- scikit-learn version mismatch (Step 0)
- missing model files in `models/`
- incorrect environment variables or startup command

---

## Where the model files go in git

Your two `.joblib` files (a Random Forest classifier + regressor,
~100 trees each) are probably a few MB — small enough to commit
directly to the repo for both platforms above. If they end up larger
than ~50MB, use [Git LFS](https://git-lfs.com) (`git lfs track "*.joblib"`)
before committing.

---

## What the app actually does (for your report / demo talk)

- `GET /` — renders the form.
- `POST /` — reads the 9 raw fields, computes `monthly_debt_est` and
  `credit_risk_index` exactly like cell 10 of your notebook, feeds a
  one-row DataFrame through both pipelines, and renders the result
  (probability of default + recommended action + suggested loan size).
- `POST /api/predict` — same logic, returns JSON. Useful if you want to
  call it from Postman, curl, or another app instead of the HTML form.
- `GET /health` — plain liveness check, handy for confirming the
  deployment actually started.
