import React from "react";
import { summary } from "../data.js";
import { Findings, PageHeader, Section } from "../components.jsx";

export default function Summary() {
  return (
    <>
      <PageHeader
        eyebrow="Conclusion"
        title="Summary of Results"
        lead="The selected model for each question, what the experiments taught us, and the limitations of this study."
      />

      <Section n="5.1" title="Selected model per question">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="left">Question</th>
                <th className="left">Selected model</th>
                <th className="left">Result</th>
                <th className="left">Note</th>
              </tr>
            </thead>
            <tbody>
              {summary.best.map((b) => (
                <tr key={b.q}>
                  <td className="left">{b.q}</td>
                  <td className="left"><strong>{b.model}</strong></td>
                  <td className="left">{b.metric}</td>
                  <td className="left">{b.alt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section n="5.2" title="Lessons">
        <Findings items={summary.lessons} />
      </Section>

      <Section n="5.3" title="Limitations and future work">
        <ul className="plain-list">
          {summary.limits.map((l) => <li key={l}>{l}</li>)}
        </ul>
      </Section>
    </>
  );
}
