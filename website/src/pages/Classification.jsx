import React, { useState } from "react";
import { q1 } from "../data.js";
import {
  BarChart, DataTable, Figure, Findings, KeyValue, ModelCard, PageHeader, Section, Segmented,
} from "../components.jsx";
import { RunClassification } from "../run.jsx";

const f4 = (v) => v.toFixed(4);

export default function Classification() {
  const [metric, setMetric] = useState("ROC-AUC");
  const chartData = q1.results.map((r) => ({ label: r.model, value: r[metric] }));
  const best = Math.max(...chartData.map((d) => d.value));

  return (
    <>
      <PageHeader
        eyebrow="Question 1 · Supervised learning"
        title="Classification: Chronic Kidney Disease"
        lead="Predict whether a patient has chronic kidney disease from 24 clinical attributes, and compare four classifiers on the same 80 held-out patients."
      />

      <Section n="1.1" title="Dataset">
        <KeyValue rows={q1.dataset} />
      </Section>

      <Section n="1.2" title="Preprocessing">
        <KeyValue rows={q1.preprocessing} />
      </Section>

      <Section n="1.3" title="Models">
        <div className="card-grid">
          {q1.models.map((m) => <ModelCard key={m.name} m={m} />)}
        </div>
      </Section>

      <Section n="Live" title="Run a classifier">
        <p className="muted small">Choose a model and its hyper-parameters; it is trained on the 320 training patients and evaluated on the same 80 test patients as the report.</p>
        <RunClassification />
      </Section>

      <Section n="1.4" title="Model comparison on the test set">
        <DataTable
          caption="80 held-out patients (50 CKD, 30 not-CKD). Bold marks the best value in each column; the shaded row is the selected model."
          highlightRow="SVM (RBF)"
          rows={q1.results}
          columns={[
            { key: "model", label: "Model", align: "left" },
            ...q1.metrics.map((m) => ({ key: m, label: m, fmt: f4, best: "max" })),
            { key: "TN", label: "TN" },
            { key: "FP", label: "FP", best: "min" },
            { key: "FN", label: "FN", best: "min" },
            { key: "TP", label: "TP", best: "max" },
          ]}
        />
        <div className="chart-controls">
          <span className="muted small">Compare by</span>
          <Segmented options={q1.metrics} value={metric} onChange={setMetric} label="Metric" />
        </div>
        <BarChart
          title={`${metric} by model`}
          data={chartData}
          domain={[0.9, 1]}
          fmt={f4}
          highlight={chartData.filter((d) => d.value === best).map((d) => d.label)}
          note="Axis runs from 0.90 to 1.00 so that small differences are visible."
        />
      </Section>

      <Section n="1.5" title="Graphs">
        <div className="figure-grid">
          <Figure src="four_model_roc_curves.png" caption="ROC curves of all four models on the test set" />
          <Figure src="confusion_matrices.png" caption="Confusion matrices: KNN, Decision Tree, SVM" />
          <Figure src="overfitting_knn.png" caption="KNN: training vs 5-fold CV accuracy as K grows" />
          <Figure src="overfitting_svm.png" caption="SVM: accuracy vs γ at C = 0.1 (log scale)" />
          <Figure src="overfitting_decision_tree.png" caption="Decision Tree: training vs test accuracy by depth" />
          <Figure src="cnn_training_curves.png" caption="1D CNN: training and validation accuracy and loss" />
        </div>
      </Section>

      <Section n="1.6" title="Decision tree depth analysis">
        <DataTable
          rowKey="depth"
          rows={q1.depth.map((r) => ({ ...r, gap: r.train - r.test }))}
          columns={[
            { key: "depth", label: "max_depth", align: "left" },
            { key: "train", label: "Train accuracy", fmt: f4 },
            { key: "test", label: "Test accuracy", fmt: f4, best: "max" },
            { key: "gap", label: "Gap (train − test)", fmt: f4 },
          ]}
        />
        <p className="muted small">The train–test gap stays between 0.006 and 0.016: more capacity, but no severe overfitting on this split.</p>
      </Section>

      <Section n="1.7" title="Conclusion">
        <Findings items={q1.findings} />
      </Section>
    </>
  );
}
