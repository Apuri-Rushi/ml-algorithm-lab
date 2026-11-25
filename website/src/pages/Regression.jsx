import React, { useState } from "react";
import { q2 } from "../data.js";
import {
  BarChart, DataTable, Figure, Findings, KeyValue, ModelCard, PageHeader, Section, Segmented,
} from "../components.jsx";
import { RunRegression } from "../run.jsx";

const f5 = (v) => (v < 0 ? "−" + Math.abs(v).toFixed(5) : v.toFixed(5));

export default function Regression() {
  const [view, setView] = useState("Test MSE");
  const ranked = [...q2.results].sort((a, b) => a.testMSE - b.testMSE);
  const chart =
    view === "Test MSE"
      ? {
          data: ranked.map((r) => ({ label: r.model, value: r.testMSE })),
          domain: [0.001, 1], log: true, fmt: (v) => v.toFixed(5),
          note: "Logarithmic axis: degree 4's test MSE is about 190 times that of degree 2.",
        }
      : {
          data: ranked.filter((r) => r.testR2 > 0).map((r) => ({ label: r.model, value: r.testR2 })),
          domain: [0.5, 0.85], log: false, fmt: (v) => v.toFixed(3),
          note: "Polynomial degree 4 is omitted: its test R² is −32.97 (worse than predicting the mean). Axis starts at 0.50.",
        };

  return (
    <>
      <PageHeader
        eyebrow="Question 2 · Supervised learning"
        title="Regression: Graduate Admissions"
        lead="Predict an applicant's chance of admission from test scores and profile, comparing a hand-written gradient descent model with polynomial and regularised regressions."
      />

      <Section n="2.1" title="Dataset">
        <KeyValue rows={q2.dataset} />
      </Section>

      <Section n="2.2" title="Models">
        <div className="card-grid">
          {q2.models.map((m) => <ModelCard key={m.name} m={m} />)}
        </div>
      </Section>

      <Section n="Live" title="Run a regression model">
        <p className="muted small">Same 320 / 80 split as the report. Try a large learning rate to see gradient descent diverge, or degree 4 to see overfitting.</p>
        <RunRegression />
      </Section>

      <Section n="2.3" title="Model comparison">
        <DataTable
          caption="Ranked by test MSE (lower is better). Bold marks the best value in each column; the shaded row is the selected model."
          highlightRow="Polynomial d = 2"
          rows={ranked}
          columns={[
            { key: "model", label: "Model", align: "left" },
            { key: "trainMSE", label: "Train MSE", fmt: f5 },
            { key: "testMSE", label: "Test MSE", fmt: f5, best: "min" },
            { key: "testRMSE", label: "Test RMSE", fmt: f5, best: "min" },
            { key: "trainR2", label: "Train R²", fmt: f5 },
            { key: "testR2", label: "Test R²", fmt: f5, best: "max" },
          ]}
        />
        <div className="chart-controls">
          <span className="muted small">Compare by</span>
          <Segmented options={["Test MSE", "Test R²"]} value={view} onChange={setView} label="Metric" />
        </div>
        <BarChart title={`${view} by model`} highlight="Polynomial d = 2" {...chart} />
      </Section>

      <Section n="2.4" title="Underfitting, good fit and overfitting">
        <div className="fit-grid">
          {q2.fit.map((f) => (
            <div className="card fit-card" key={f.label}>
              <p className="eyebrow">{f.label}</p>
              <h3>{f.model}</h3>
              <p className="big-number">{f.testR2}</p>
              <p className="muted small">Test R² · train R² {f.trainR2}</p>
              <p className="small">{f.text}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section n="2.5" title="Learned weights (gradient descent, standardised inputs)">
        <BarChart
          data={q2.weights.map(([label, value]) => ({ label, value: Math.max(value, 0) }))}
          domain={[0, 0.07]}
          fmt={(v) => (v === 0 ? "−0.0003" : v.toFixed(4))}
          highlight="CGPA"
          note="CGPA dominates; SOP contributes almost nothing once the other features are known. Weights are associations, not causal effects."
        />
      </Section>

      <Section n="2.6" title="Graphs">
        <div className="figure-grid">
          <Figure src="regression_gd_mse_vs_epochs.png" caption="Gradient descent: training MSE over 5,000 epochs" />
          <Figure src="regression_linear_actual_vs_predicted.png" caption="Linear (GD): actual vs predicted" />
          <Figure src="regression_polynomial_degrees_error.png" caption="Training vs test error by polynomial degree" />
          <Figure src="regression_polynomial_degrees_scatter_grid.png" caption="Actual vs predicted for degrees 1–4" />
          <Figure src="regression_ridge_lasso_cv_alpha.png" caption="Ridge and Lasso: cross-validation MSE vs α" />
          <Figure src="regression_poly2_residuals.png" caption="Degree-2 polynomial: residuals" />
        </div>
      </Section>

      <Section n="2.7" title="Conclusion">
        <Findings items={q2.findings} />
      </Section>
    </>
  );
}
