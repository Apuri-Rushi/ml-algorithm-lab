import React, { useEffect, useState } from "react";

export function PageHeader({ eyebrow, title, lead }) {
  return (
    <header className="page-header">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      {lead && <p className="lead">{lead}</p>}
    </header>
  );
}

export function Section({ n, title, children }) {
  return (
    <section className="section">
      <h2>
        {n && <span className="sec-num">{n}</span>}
        {title}
      </h2>
      {children}
    </section>
  );
}

export function KeyValue({ rows }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div className="kv-row" key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Steps({ items }) {
  return (
    <ol className="steps">
      {items.map(([title, text, formula]) => (
        <li key={title}>
          <strong>{title}.</strong> {text}
          {formula && <code className="formula inline">{formula}</code>}
        </li>
      ))}
    </ol>
  );
}

export function ModelCard({ m }) {
  return (
    <article className="card model-card">
      <div className="card-head">
        <h3>{m.name}</h3>
        {m.cv && (
          <div className="card-stat">
            <span>{m.cvLabel}</span>
            <strong>{m.cv}</strong>
          </div>
        )}
      </div>
      <p>{m.theory}</p>
      <code className="formula">{m.formula}</code>
      {m.grid && (
        <p className="muted small">
          <span className="label">Search</span> {m.grid}
        </p>
      )}
      <p className="selected">
        <span className="label">Selected</span> {m.best}
      </p>
    </article>
  );
}

/**
 * Plain table. columns: [{ key, label, fmt?, align?, best?: "max" | "min" }]
 * The best value in a column marked with `best` is set in bold on a tinted cell.
 */
export function DataTable({ columns, rows, caption, rowKey = "model", highlightRow }) {
  const bestOf = {};
  columns.forEach((c) => {
    if (!c.best) return;
    const vals = rows.map((r) => r[c.key]).filter((v) => typeof v === "number");
    bestOf[c.key] = c.best === "max" ? Math.max(...vals) : Math.min(...vals);
  });
  return (
    <div className="table-wrap">
      <table>
        {caption && <caption>{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.align === "left" ? "left" : ""}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[rowKey]} className={highlightRow === r[rowKey] ? "row-selected" : ""}>
              {columns.map((c) => {
                const v = r[c.key];
                const isBest = c.best && v === bestOf[c.key];
                return (
                  <td key={c.key} className={`${c.align === "left" ? "left" : ""} ${isBest ? "best" : ""}`}>
                    {c.fmt ? c.fmt(v) : v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Horizontal bar chart drawn in SVG.
 * data: [{ label, value }]; domain: [min, max]; log: use a log10 scale.
 * highlight: label of the bar to draw darker.
 */
export function BarChart({ data, domain, log = false, fmt = (v) => v, highlight, title, note }) {
  const W = 760, labelW = 190, valueW = 90, barH = 24, gap = 10, top = 8;
  const H = top + data.length * (barH + gap);
  const plotW = W - labelW - valueW;
  const [lo, hi] = domain;
  const t = (v) => (log ? Math.log10(v) : v);
  const scale = (v) => {
    const x = (t(Math.min(Math.max(v, lo), hi)) - t(lo)) / (t(hi) - t(lo));
    return Math.max(2, x * plotW);
  };
  return (
    <figure className="chart">
      {title && <figcaption className="chart-title">{title}</figcaption>}
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title || "bar chart"}>
        {data.map((d, i) => {
          const y = top + i * (barH + gap);
          const w = scale(d.value);
          const strong = Array.isArray(highlight) ? highlight.includes(d.label) : d.label === highlight;
          return (
            <g key={d.label}>
              <text x={labelW - 12} y={y + barH / 2} className="chart-label" textAnchor="end" dominantBaseline="middle">
                {d.label}
              </text>
              <rect x={labelW} y={y} width={plotW} height={barH} className="chart-track" />
              <rect x={labelW} y={y} width={w} height={barH} className={strong ? "chart-bar strong" : "chart-bar"} />
              <text x={labelW + w + 8} y={y + barH / 2} className="chart-value" dominantBaseline="middle">
                {fmt(d.value)}
              </text>
            </g>
          );
        })}
      </svg>
      {note && <p className="chart-note">{note}</p>}
    </figure>
  );
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o} className={o === value ? "on" : ""} onClick={() => onChange(o)} aria-pressed={o === value}>
          {o}
        </button>
      ))}
    </div>
  );
}

export function Figure({ src, caption }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  const path = `./figures/${src}`;
  return (
    <>
      <figure className="figure">
        <button className="figure-btn" onClick={() => setOpen(true)} aria-label={`Enlarge: ${caption}`}>
          <img src={path} alt={caption} loading="lazy" />
        </button>
        <figcaption>{caption}</figcaption>
      </figure>
      {open && (
        <div className="lightbox" onClick={() => setOpen(false)} role="dialog" aria-label={caption}>
          <img src={path} alt={caption} />
          <p>{caption} · click anywhere or press Esc to close</p>
        </div>
      )}
    </>
  );
}

export function Findings({ items }) {
  return (
    <div className="findings">
      {items.map(([k, v]) => (
        <div className="finding" key={k}>
          <h4>{k}</h4>
          <p>{v}</p>
        </div>
      ))}
    </div>
  );
}
