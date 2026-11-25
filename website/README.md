# Project Results Website

A React (Vite) site that presents the four experiments question by question
(Overview · Q1 Classification · Q2 Regression · Q3 K-Means · Q4 PCA · Summary)
and can **run the real models live** through a small local Python server.

**Online version:** https://apuri-rushi.github.io/ml-algorithm-lab/
(results, theory and graphs; the live "Run" panels need the local server below).

## Run (one command)

```powershell
cd website
npm start
```

This starts the Python model server (`server.py`, port 8000) and the website
(`http://localhost:5173`, opens in the browser). Press **Ctrl+C** to stop both.

The first time only: `npm install`.

### Running the two parts separately (optional)

```powershell
python website/server.py        # terminal 1: model server
cd website; npm run dev         # terminal 2: website
```

If the website is open without the server, each "Run" panel shows a red notice
explaining how to start it; the rest of the site still works.

## Live "Run" panels

| Page | What you can change | Typical time |
|---|---|---|
| Q1 Classification | Model (KNN / Decision Tree / SVM / 1D CNN), K and distance, depth and leaf size, C and γ, a test patient, optional grid search | < 1 s; grid search 10–30 s; CNN ~1 min |
| Q2 Regression | Gradient descent learning rate and epochs, polynomial degree 1–5, Ridge / Lasso α (or CV) | < 2 s |
| Q3 K-Means | K = 2 … 8 (from-scratch NumPy implementation, 10 restarts) | 3–8 s |
| Q4 PCA | Number of components 1–4 (manual eigen-decomposition, checked against sklearn) | < 1 s |

The server reuses the functions in `demo/` and the same data splits, so default settings reproduce the report's numbers.

## Where things are

| File | Contents |
|---|---|
| `server.py` | Local model server (Python standard library + scikit-learn / NumPy / TensorFlow) |
| `start.mjs` | Starts server and website together (`npm start`) |
| `src/run.jsx` | The live "Run" panels |
| `src/data.js` | Every saved number shown on the site (from the project's result files) |
| `src/pages/*.jsx` | One page per question |
| `src/components.jsx` | Tables, bar charts, model cards, figures |
| `src/styles.css` | Formal charcoal theme |
| `public/figures/` | Saved graphs from the project |

`server.py` imports from `../demo/`, so the `demo/` folder must stay next to `website/`.
