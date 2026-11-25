import React, { useEffect, useState } from "react";

/* ------------------------------------------------------------------ plumbing */
const SERVER_HINT = "The model server is not running. Open a terminal in the project folder and run:  python website/server.py";

function useRunner(endpoint) {
  const [state, setState] = useState({ loading: false, error: null, result: null });
  const run = async (params) => {
    setState({ loading: true, error: null, result: null });
    try {
      const res = await fetch(`/api/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      let data;
      try { data = await res.json(); } catch { throw new Error(SERVER_HINT); }
      if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
      setState({ loading: false, error: null, result: data });
    } catch (e) {
      const msg = e instanceof TypeError ? SERVER_HINT : e.message;
      setState({ loading: false, error: msg, result: null });
    }
  };
  return [state, run];
}

function ServerStatus() {
  const [ok, setOk] = useState(null);
  useEffect(() => {
    let alive = true;
    const check = () =>
      fetch("/api/health").then((r) => r.json()).then(() => alive && setOk(true)).catch(() => alive && setOk(false));
    check();
    const id = setInterval(check, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  if (ok === null) return null;
  return (
    <p className={`server-status ${ok ? "on" : "off"}`}>
      <span className="dot" aria-hidden="true" />
      {ok ? "Model server connected — models run live on this computer." : SERVER_HINT}
    </p>
  );
}

function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

function Select({ value, onChange, options }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => {
        const [v, l] = Array.isArray(o) ? o : [o, o];
        return <option key={v} value={v}>{l}</option>;
      })}
    </select>
  );
}

function NumberInput({ value, onChange, min, max, step = 1 }) {
  return <input type="number" value={value} min={min} max={max} step={step} onChange={(e) => onChange(e.target.value)} />;
}

function ResultHeader({ r }) {
  return (
    <div className="result-head">
      <h3>{r.model}</h3>
      <span className="muted small">Computed in {r.seconds} s</span>
    </div>
  );
}

function Notes({ notes }) {
  if (!notes || !notes.length) return null;
  return <ul className="run-notes">{notes.map((n) => <li key={n}>{n}</li>)}</ul>;
}

function MiniTable({ head, rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{head.map((h, i) => <th key={h} className={i === 0 ? "left" : ""}>{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={r.__selected ? "row-selected" : ""}>
              {r.cells.map((c, j) => <td key={j} className={j === 0 ? "left" : ""}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Plot({ b64, alt }) {
  if (!b64) return null;
  return <img className="run-plot" src={`data:image/png;base64,${b64}`} alt={alt} />;
}

const f = (v, n = 4) => (v == null ? "—" : Number(v).toFixed(n));

/* ------------------------------------------------------------------ Q1 */
export function RunClassification() {
  const [p, setP] = useState({ model: "svm", k: 9, metric: "manhattan", depth: "3", leaf: 5, C: "0.1", gamma: "scale", tune: false, patient: "" });
  const [state, run] = useRunner("q1");
  const set = (k) => (v) => setP({ ...p, [k]: v });
  const r = state.result;
  const isCnn = p.model === "cnn";

  return (
    <div className="run-panel">
      <ServerStatus />
      <div className="run-form">
        <Field label="Model">
          <Select value={p.model} onChange={set("model")} options={[["knn", "K-Nearest Neighbours"], ["tree", "Decision Tree"], ["svm", "SVM (RBF)"], ["cnn", "1D CNN (bonus)"]]} />
        </Field>
        {p.model === "knn" && <>
          <Field label="K (neighbours)"><NumberInput value={p.k} onChange={set("k")} min={1} max={25} /></Field>
          <Field label="Distance"><Select value={p.metric} onChange={set("metric")} options={["manhattan", "euclidean"]} /></Field>
        </>}
        {p.model === "tree" && <>
          <Field label="max_depth"><Select value={p.depth} onChange={set("depth")} options={["1", "2", "3", "5", "10", "None"]} /></Field>
          <Field label="min_samples_leaf"><NumberInput value={p.leaf} onChange={set("leaf")} min={1} max={30} /></Field>
        </>}
        {p.model === "svm" && <>
          <Field label="C"><Select value={p.C} onChange={set("C")} options={["0.01", "0.1", "1", "10", "100"]} /></Field>
          <Field label="γ (gamma)"><Select value={p.gamma} onChange={set("gamma")} options={["scale", "0.001", "0.01", "0.1", "1", "10"]} /></Field>
        </>}
        {!isCnn && <>
          <Field label="Test patient #" hint="0–79, optional"><NumberInput value={p.patient} onChange={set("patient")} min={0} max={79} /></Field>
          <label className="check">
            <input type="checkbox" checked={p.tune} onChange={(e) => set("tune")(e.target.checked)} />
            Tune with grid search (5-fold CV, ~10–30 s)
          </label>
        </>}
        {isCnn && <p className="field-hint wide">Trains the network from scratch: about one minute.</p>}
        <button className="run-btn" onClick={() => run(p)} disabled={state.loading}>
          {state.loading ? "Running…" : "Run model"}
        </button>
      </div>
      {state.error && <p className="run-error" role="alert">{state.error}</p>}
      {state.loading && <p className="muted small">Training on 320 patients and testing on 80 with the project's Python code…</p>}
      {r && <>
        <ResultHeader r={r} />
        {r.tuning && <p className="small"><span className="label">Grid search</span> best {Object.entries(r.tuning.best_params).map(([k, v]) => `${k} = ${v}`).join(", ")} · CV ROC-AUC {f(r.tuning.cv_auc)}</p>}
        <MiniTable
          head={["Test set (80)", ...Object.keys(r.metrics), "TN", "FP", "FN", "TP"]}
          rows={[{ cells: [r.model, ...Object.values(r.metrics).map((v) => f(v)), r.confusion.TN, r.confusion.FP, r.confusion.FN, r.confusion.TP] }]}
        />
        {r.train_accuracy != null && <p className="small muted">Training accuracy {f(r.train_accuracy)} vs test accuracy {f(r.metrics.Accuracy)}.</p>}
        {r.patient && (
          <div className="patient">
            <p><span className="label">Patient #{r.patient.index}</span> P(CKD) = <strong>{r.patient.p_ckd}</strong> → predicted <strong>{r.patient.predicted}</strong>; actual <strong>{r.patient.true}</strong></p>
            <p className="small muted">{Object.entries(r.patient.features).map(([k, v]) => `${k} ${v ?? "missing"}`).join(" · ")}</p>
          </div>
        )}
        <Notes notes={r.notes} />
        <Plot b64={r.image} alt="ROC curve and confusion matrix" />
      </>}
    </div>
  );
}

/* ------------------------------------------------------------------ Q2 */
export function RunRegression() {
  const [p, setP] = useState({ mode: "gd", lr: 0.05, epochs: 5000, degree: "2", alpha: "cv" });
  const [state, run] = useRunner("q2");
  const set = (k) => (v) => setP({ ...p, [k]: v });
  const r = state.result;

  return (
    <div className="run-panel">
      <ServerStatus />
      <div className="run-form">
        <Field label="Model">
          <Select value={p.mode} onChange={set("mode")} options={[["gd", "Linear — gradient descent"], ["poly", "Polynomial"], ["ridge", "Ridge (L2)"], ["lasso", "Lasso (L1)"]]} />
        </Field>
        {p.mode === "gd" && <>
          <Field label="Learning rate η" hint="stable below ≈ 0.206"><NumberInput value={p.lr} onChange={set("lr")} min={0.0001} max={2} step={0.01} /></Field>
          <Field label="Epochs"><NumberInput value={p.epochs} onChange={set("epochs")} min={1} max={50000} step={500} /></Field>
        </>}
        {p.mode === "poly" && (
          <Field label="Degree"><Select value={p.degree} onChange={set("degree")} options={["1", "2", "3", "4", "5"]} /></Field>
        )}
        {(p.mode === "ridge" || p.mode === "lasso") && (
          <Field label="α (alpha)">
            <Select value={p.alpha} onChange={set("alpha")} options={[["cv", "Choose by 5-fold CV"], "0.0001", "0.001", "0.01", "0.1", "1", "10", "100", ["1000000", "1,000,000 (underfit)"]]} />
          </Field>
        )}
        <button className="run-btn" onClick={() => run(p)} disabled={state.loading}>
          {state.loading ? "Running…" : "Run model"}
        </button>
      </div>
      {state.error && <p className="run-error" role="alert">{state.error}</p>}
      {r && <>
        <ResultHeader r={r} />
        {r.test && (
          <MiniTable
            head={["Partition", "MSE", "RMSE", "R²"]}
            rows={[
              { cells: ["Train (320)", f(r.train.MSE, 5), f(r.train.RMSE, 5), f(r.train["R²"], 5)] },
              { cells: ["Test (80)", f(r.test.MSE, 5), f(r.test.RMSE, 5), f(r.test["R²"], 5)], __selected: true },
            ]}
          />
        )}
        <Notes notes={r.notes} />
        {r.weights && (
          <details className="weights">
            <summary>Learned coefficients</summary>
            <MiniTable head={["Feature", "Weight"]} rows={Object.entries(r.weights).map(([k, v]) => ({ cells: [k, v] }))} />
          </details>
        )}
        <Plot b64={r.image} alt="Regression result graph" />
      </>}
    </div>
  );
}

/* ------------------------------------------------------------------ Q3 */
export function RunKMeans() {
  const [k, setK] = useState("4");
  const [state, run] = useRunner("q3");
  const r = state.result;
  return (
    <div className="run-panel">
      <ServerStatus />
      <div className="run-form">
        <Field label="Number of clusters K"><Select value={k} onChange={setK} options={["2", "3", "4", "5", "6", "7", "8"]} /></Field>
        <button className="run-btn" onClick={() => run({ k })} disabled={state.loading}>
          {state.loading ? "Clustering 17,379 hours…" : "Run K-Means"}
        </button>
      </div>
      {state.error && <p className="run-error" role="alert">{state.error}</p>}
      {r && <>
        <ResultHeader r={r} />
        <p className="small"><span className="label">WCSS</span> {r.wcss.toLocaleString("en-US")}</p>
        <MiniTable
          head={["Cluster", "Hours", "Share", "Temp", "Humidity", "Wind", "Rentals / hour"]}
          rows={r.clusters.map((c) => ({ cells: [c.cluster, c.hours.toLocaleString("en-US"), `${c.share} %`, c.temp.toFixed(2), c.hum.toFixed(2), c.wind.toFixed(2), c.rentals.toFixed(1)] }))}
        />
        <Notes notes={r.notes} />
        <Plot b64={r.image} alt="Cluster scatter and demand by cluster" />
      </>}
    </div>
  );
}

/* ------------------------------------------------------------------ Q4 */
export function RunPCA() {
  const [n, setN] = useState("2");
  const [state, run] = useRunner("q4");
  const r = state.result;
  return (
    <div className="run-panel">
      <ServerStatus />
      <div className="run-form">
        <Field label="Components to keep"><Select value={n} onChange={setN} options={["1", "2", "3", "4"]} /></Field>
        <button className="run-btn" onClick={() => run({ n })} disabled={state.loading}>
          {state.loading ? "Running…" : "Run PCA"}
        </button>
      </div>
      {state.error && <p className="run-error" role="alert">{state.error}</p>}
      {r && <>
        <ResultHeader r={r} />
        <div className="stat-row">
          <div className="stat"><strong>{r.retained} %</strong><span>variance retained · {r.shape}</span></div>
          <div className="stat"><strong>{f(r.reconstruction_mse, 5)}</strong><span>reconstruction MSE (standardised)</span></div>
        </div>
        <MiniTable
          head={["Component", "Eigenvalue", "Variance", "Cumulative", "Kept"]}
          rows={r.components.map((c) => ({ cells: [c.pc, f(c.eigenvalue), `${c.variance.toFixed(2)} %`, `${c.cumulative.toFixed(2)} %`, c.kept ? "Yes" : "—"], __selected: c.kept }))}
        />
        <Notes notes={r.notes} />
        <Plot b64={r.image} alt="PCA projection and explained variance" />
      </>}
    </div>
  );
}
