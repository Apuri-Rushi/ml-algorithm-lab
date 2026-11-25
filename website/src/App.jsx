import React, { useEffect, useState } from "react";
import { project } from "./data.js";
import Overview from "./pages/Overview.jsx";
import Classification from "./pages/Classification.jsx";
import Regression from "./pages/Regression.jsx";
import Clustering from "./pages/Clustering.jsx";
import PCA from "./pages/PCA.jsx";
import Summary from "./pages/Summary.jsx";

const PAGES = [
  { id: "overview", label: "Overview", short: "Overview", el: Overview },
  { id: "q1", label: "Q1 · Classification", short: "Q1", el: Classification },
  { id: "q2", label: "Q2 · Regression", short: "Q2", el: Regression },
  { id: "q3", label: "Q3 · K-Means", short: "Q3", el: Clustering },
  { id: "q4", label: "Q4 · PCA", short: "Q4", el: PCA },
  { id: "summary", label: "Summary", short: "Summary", el: Summary },
];

const fromHash = () => {
  const h = window.location.hash.replace("#", "");
  return PAGES.some((p) => p.id === h) ? h : "overview";
};

export default function App() {
  const [page, setPage] = useState(fromHash);

  useEffect(() => {
    const onHash = () => {
      setPage(fromHash());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const idx = PAGES.findIndex((p) => p.id === page);
  const Current = PAGES[idx].el;
  const prev = PAGES[idx - 1];
  const next = PAGES[idx + 1];

  return (
    <div className="app">
      <header className="site-header">
        <div className="wrap header-inner">
          <a href="#overview" className="brand">
            <span className="brand-inst">{project.institute}</span>
            <span className="brand-title">{project.title}</span>
          </a>
          <p className="authors">
            {project.authors.map((a) => a.name).join(" · ")} · {project.section}
          </p>
        </div>
        <nav className="site-nav" aria-label="Sections">
          <div className="wrap nav-inner">
            {PAGES.map((p) => (
              <a key={p.id} href={`#${p.id}`} className={p.id === page ? "active" : ""} aria-current={p.id === page ? "page" : undefined}>
                <span className="nav-long">{p.label}</span>
                <span className="nav-short">{p.short}</span>
              </a>
            ))}
          </div>
        </nav>
      </header>

      <main className="wrap main">
        <Current />
        <nav className="pager" aria-label="Page navigation">
          {prev ? <a href={`#${prev.id}`}>← {prev.label}</a> : <span />}
          {next ? <a href={`#${next.id}`}>{next.label} →</a> : <span />}
        </nav>
      </main>

      <footer className="site-footer">
        <div className="wrap">
          {project.authors.map((a) => `${a.name} (${a.id})`).join(" · ")} — {project.institute}
        </div>
      </footer>
    </div>
  );
}
