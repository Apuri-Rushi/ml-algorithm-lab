import React from "react";
import { q4 } from "../data.js";
import { BarChart, DataTable, Figure, Findings, KeyValue, PageHeader, Section, Steps } from "../components.jsx";
import { RunPCA } from "../run.jsx";

const sgn = (v) => (v < 0 ? "−" + Math.abs(v).toFixed(5) : v.toFixed(5));

export default function PCA() {
  return (
    <>
      <PageHeader
        eyebrow="Question 4 · Unsupervised learning"
        title="Principal Component Analysis: Iris"
        lead="Reduce four flower measurements to fewer dimensions by eigen-decomposing the covariance matrix, implemented manually in NumPy."
      />

      <Section n="4.1" title="Dataset">
        <KeyValue rows={q4.dataset} />
      </Section>

      <Section n="4.2" title="Method">
        <Steps items={q4.steps} />
        <p className="muted small">
          The covariance diagonal is 1.00671 rather than 1 because np.cov divides by n − 1 while the scaler divides by n
          (150 / 149). Variance ratios are unaffected.
        </p>
      </Section>

      <Section n="Live" title="Run PCA">
        <p className="muted small">Runs the manual covariance–eigenvector PCA and checks it against scikit-learn.</p>
        <RunPCA />
      </Section>

      <Section n="4.3" title="Explained variance by component">
        <div className="two-col">
          <DataTable
            rowKey="pc"
            rows={q4.components}
            columns={[
              { key: "pc", label: "Component", align: "left" },
              { key: "eig", label: "Eigenvalue", fmt: (v) => v.toFixed(4) },
              { key: "ratio", label: "Variance", fmt: (v) => `${v.toFixed(2)} %` },
              { key: "cum", label: "Cumulative", fmt: (v) => `${v.toFixed(2)} %` },
            ]}
          />
          <BarChart
            data={q4.components.map((c) => ({ label: c.pc, value: c.ratio }))}
            domain={[0, 80]}
            fmt={(v) => `${v.toFixed(2)} %`}
            highlight="PC1"
          />
        </div>
        <div className="stat-row">
          <div className="stat"><strong>95.81 %</strong><span>2 components · 150 × 2</span></div>
          <div className="stat"><strong>99.48 %</strong><span>3 components · 150 × 3</span></div>
        </div>
      </Section>

      <Section n="4.4" title="Eigenvectors (loadings)">
        <DataTable
          rowKey="0"
          rows={q4.eigenvectors.map((r) => ({ 0: r[0], 1: r[1], 2: r[2], 3: r[3], 4: r[4] }))}
          columns={[
            { key: "0", label: "Feature", align: "left" },
            { key: "1", label: "PC1", fmt: sgn },
            { key: "2", label: "PC2", fmt: sgn },
            { key: "3", label: "PC3", fmt: sgn },
            { key: "4", label: "PC4", fmt: sgn },
          ]}
        />
        <p className="muted small">An eigenvector and its negative describe the same axis, so another correct implementation may show flipped signs.</p>
      </Section>

      <Section n="4.5" title="Graphs">
        <div className="figure-grid">
          <Figure src="pca_scatter_pc1_pc2.png" caption="Iris projected onto PC1 and PC2, coloured by species" />
          <Figure src="cumulative_explained_variance.png" caption="Cumulative explained variance" />
        </div>
      </Section>

      <Section n="4.6" title="Conclusion">
        <Findings items={q4.findings} />
      </Section>
    </>
  );
}
