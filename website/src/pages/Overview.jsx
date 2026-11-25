import React from "react";
import { overview, project, protocol } from "../data.js";
import { PageHeader, Section } from "../components.jsx";

export default function Overview() {
  return (
    <>
      <PageHeader
        eyebrow="ML Course Project · Overview"
        title={project.title}
        lead="Four experiments on four datasets: two supervised tasks (classification and regression) and two unsupervised tasks (clustering and dimensionality reduction). Linear regression by gradient descent, K-Means and PCA were implemented from scratch."
      />

      <Section n="1" title="Experiments at a glance">
        <div className="overview-grid">
          {overview.map((o) => (
            <a className="card overview-card" href={`#${o.id}`} key={o.id}>
              <div className="ov-top">
                <span className="ov-q">{o.q}</span>
                <span className="tag">{o.kind}</span>
              </div>
              <h3>{o.task}</h3>
              <p className="muted">{o.dataset}</p>
              <p className="ov-result">{o.result}</p>
              <p className="small">{o.note}</p>
              <hr />
              <p className="small"><span className="label">Size</span> {o.size}</p>
              <p className="small"><span className="label">Models</span> {o.models}</p>
              <span className="ov-link">View results →</span>
            </a>
          ))}
        </div>
      </Section>

      <Section n="2" title="Experimental protocol">
        <ul className="plain-list">
          {protocol.map((p) => <li key={p}>{p}</li>)}
        </ul>
      </Section>

      <Section n="3" title="Comparison at a glance">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="left">Question</th>
                <th className="left">Dataset</th>
                <th className="left">Models compared</th>
                <th className="left">Selected result</th>
              </tr>
            </thead>
            <tbody>
              {overview.map((o) => (
                <tr key={o.id}>
                  <td className="left"><a href={`#${o.id}`}>{o.q} · {o.task}</a></td>
                  <td className="left">{o.dataset}</td>
                  <td className="left">{o.models}</td>
                  <td className="left"><strong>{o.result}</strong> — {o.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}
