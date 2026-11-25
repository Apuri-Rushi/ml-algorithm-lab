import React from "react";
import { q3 } from "../data.js";
import { BarChart, DataTable, Figure, Findings, KeyValue, PageHeader, Section, Steps } from "../components.jsx";
import { RunKMeans } from "../run.jsx";

const n2 = (v) => v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export default function Clustering() {
  return (
    <>
      <PageHeader
        eyebrow="Question 3 · Unsupervised learning"
        title="K-Means Clustering: Bike Sharing"
        lead="Group hourly bike-rental records by time of day and weather using K-Means written from scratch in NumPy, then describe each group."
      />

      <Section n="3.1" title="Dataset and features">
        <KeyValue rows={q3.dataset} />
      </Section>

      <Section n="3.2" title="Algorithm (implemented from scratch)">
        <Steps items={q3.steps} />
        <code className="formula block">WCSS = Σₖ Σ_(i ∈ Cₖ) ‖xᵢ − μₖ‖²</code>
        <p className="muted small">The mean is the optimal centroid because setting ∂WCSS/∂μₖ = 0 gives μₖ = (1/|Cₖ|) Σ xᵢ.</p>
      </Section>

      <Section n="Live" title="Run K-Means">
        <p className="muted small">Runs the from-scratch NumPy implementation on all 17,379 hours (10 restarts, about 3–8 seconds).</p>
        <RunKMeans />
      </Section>

      <Section n="3.3" title="Choosing K: elbow method">
        <div className="two-col">
          <DataTable
            rowKey="k"
            highlightRow={4}
            rows={q3.elbow}
            columns={[
              { key: "k", label: "K", align: "left" },
              { key: "wcss", label: "WCSS", fmt: n2 },
              { key: "drop", label: "Drop from K − 1", fmt: (v) => (v == null ? "—" : `${v.toFixed(1)} %`) },
            ]}
          />
          <BarChart
            data={q3.elbow.map((e) => ({ label: `K = ${e.k}`, value: e.wcss }))}
            domain={[0, 70000]}
            fmt={(v) => Math.round(v).toLocaleString("en-US")}
            highlight="K = 4"
          />
        </div>
        <p className="muted small">WCSS always falls as K grows; the gain shrinks clearly after K = 4.</p>
      </Section>

      <Section n="3.4" title="The four clusters (K = 4)">
        <DataTable
          caption="Weather values are the original normalised scales. Rentals per hour were computed afterwards; demand was never a clustering input."
          rowKey="id"
          highlightRow={1}
          rows={q3.clusters}
          columns={[
            { key: "id", label: "Cluster", align: "left" },
            { key: "name", label: "Profile", align: "left" },
            { key: "hours", label: "Hours", fmt: (v) => v.toLocaleString("en-US") },
            { key: "share", label: "Share", fmt: (v) => `${v.toFixed(1)} %` },
            { key: "temp", label: "Temp", fmt: (v) => v.toFixed(2), best: "max" },
            { key: "hum", label: "Humidity", fmt: (v) => v.toFixed(2), best: "max" },
            { key: "wind", label: "Wind", fmt: (v) => v.toFixed(2), best: "max" },
            { key: "rentals", label: "Rentals / hour", fmt: (v) => v.toFixed(1), best: "max" },
          ]}
        />
        <BarChart
          title="Mean rentals per hour by cluster"
          data={q3.clusters.map((c) => ({ label: `${c.id} · ${c.name}`, value: c.rentals }))}
          domain={[0, 400]}
          fmt={(v) => v.toFixed(1)}
          highlight="1 · Warm and dry"
        />
      </Section>

      <Section n="3.5" title="Graphs">
        <div className="figure-grid">
          <Figure src="kmeans_elbow_curve.png" caption="Elbow curve: WCSS for K = 2 to 6" />
          <Figure src="kmeans_clusters_with_centroids.png" caption="Clusters in the temperature–humidity plane (2 of 5 dimensions)" />
          <Figure src="kmeans_centroid_profiles.png" caption="Standardised centroid values for each feature" />
        </div>
      </Section>

      <Section n="3.6" title="Conclusion">
        <Findings items={q3.findings} />
      </Section>
    </>
  );
}
