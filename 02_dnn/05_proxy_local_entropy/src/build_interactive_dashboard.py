"""Build interactive local-entropy dashboards from retained summary CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


RUNS = [
    {
        "id": "18_beta_cell_30_dataset_30_reference",
        "label": "18 beta / 30 dataset / 30 reference",
        "range": "d_0.01_to_2.50_dense",
        "mode": "production",
    },
    {
        "id": "18_beta_cell_60_dataset_30_reference",
        "label": "18 beta / 60 dataset / 30 reference",
        "range": "d_0.01_to_2.50_dense",
        "mode": "summary_only_extension",
    },
]

TABLE_FILES = {
    "absolute": "absolute_phi_by_beta_radius.csv",
    "delta": "delta_phi_by_beta_radius.csv",
    "derivative": "dphi_dr_by_beta_radius.csv",
    "hq": "hq_by_beta_radius.csv",
}

METRICS = {
    "absolute": [
        ["phi_full", "phi full"],
        ["phi_energy", "phi energy"],
        ["phi_entropic", "phi entropic = full - energy"],
        ["phi_stripped_proxy", "phi stripped proxy"],
        ["phi_energy_stripped_proxy", "phi energy stripped proxy"],
        ["area_term", "L2 shell area term"],
        ["reference_prior_correction_per_P", "reference prior correction / P"],
    ],
    "delta": [
        ["delta_phi_full", "delta phi full"],
        ["delta_phi_energy", "delta phi energy"],
        ["delta_phi_entropic", "delta phi entropic = full - energy"],
        ["delta_phi_stripped_proxy", "delta phi stripped proxy"],
        ["delta_phi_energy_stripped", "delta phi energy stripped"],
        ["area_term", "L2 shell area term"],
    ],
    "derivative": [
        ["dphi_full_dr", "d phi full / dr"],
        ["dphi_energy_dr", "d phi energy / dr"],
        ["dphi_entropic_dr", "d phi entropic / dr"],
        ["mean_dlogZ_inf_full_dr", "mean d logZ / dr"],
        ["sd_dlogZ_inf_full_dr", "sd d logZ / dr"],
    ],
    "hq": [
        ["H_q_numeric", "H quantile"],
        ["mean_R_H_at_H_q", "mean R_H at H_q"],
        ["ref_count", "reference count"],
    ],
}


def parse_number(value: str):
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number):
        return number
    return None


def read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {key: parse_number(value) for key, value in row.items()}
            if "phi_full" in parsed and "phi_energy" in parsed:
                parsed["phi_entropic"] = parsed["phi_full"] - parsed["phi_energy"]
            if "delta_phi_full" in parsed and "delta_phi_energy" in parsed:
                parsed["delta_phi_entropic"] = (
                    parsed["delta_phi_full"] - parsed["delta_phi_energy"]
                )
            rows.append(parsed)
    return rows


def available_metrics(rows_by_table: dict[str, list[dict]]) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    for table, candidates in METRICS.items():
        rows = rows_by_table.get(table) or []
        keys = set(rows[0].keys()) if rows else set()
        out[table] = [item for item in candidates if item[0] in keys]
    return out


def load_run(repo_root: Path, run: dict) -> dict:
    summary_root = (
        repo_root
        / "02_dnn/05_proxy_local_entropy/raw_outputs"
        / run["id"]
        / run["range"]
        / "summary_tables"
    )
    tables: dict[str, list[dict]] = {}
    for table, name in TABLE_FILES.items():
        path = summary_root / name
        tables[table] = read_csv(path) if path.exists() else []

    betas = sorted(
        {
            float(row["beta"])
            for rows in tables.values()
            for row in rows
            if isinstance(row.get("beta"), (int, float))
        }
    )
    radii = sorted(
        {
            float(row["radius"])
            for rows in tables.values()
            for row in rows
            if isinstance(row.get("radius"), (int, float))
        }
    )
    qs = sorted(
        {
            float(row["q"])
            for row in tables.get("hq", [])
            if isinstance(row.get("q"), (int, float))
        }
    )
    return {
        **run,
        "tables": tables,
        "metrics": available_metrics(tables),
        "betas": betas,
        "radii": radii,
        "qs": qs,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Proxy Local Entropy Dashboard</title>
  <script src="{plotly_src}"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #596574;
      --line: #d8dde5;
      --accent: #0f766e;
      --accent-soft: #d9f2ee;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 20px 28px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      max-width: 1120px;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
    }}
    aside, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    aside {{
      padding: 14px;
      align-self: start;
      position: sticky;
      top: 12px;
      max-height: calc(100vh - 24px);
      overflow: auto;
    }}
    .control {{
      margin: 0 0 14px;
    }}
    label {{
      display: block;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    select, input, button {{
      width: 100%;
      font: inherit;
      font-size: 13px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #fff;
      color: var(--ink);
      padding: 7px 8px;
    }}
    select[multiple] {{
      min-height: 150px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .buttons {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 6px;
      margin-top: 6px;
    }}
    button {{
      cursor: pointer;
      background: #fff;
    }}
    button.primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .chart-panel {{
      padding: 12px;
      min-width: 0;
    }}
    #plot {{
      width: 100%;
      height: 72vh;
      min-height: 520px;
    }}
    #summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 0 0 12px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 9px;
      background: #fbfcfd;
    }}
    .stat .k {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .stat .v {{
      font-size: 16px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .note {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
      margin-top: 10px;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ position: static; max-height: none; }}
      #summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      #plot {{ height: 64vh; min-height: 420px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Proxy Local Entropy Dashboard</h1>
    <div class="subtitle">
      Interactive view for the retained 18-beta proxy-local-entropy runs.
      Use the controls to switch run, metric family, beta cells, radius window,
      x scale, y transform, and line/heatmap view.
    </div>
  </header>
  <main>
    <aside>
      <div class="control">
        <label for="runSelect">Runs</label>
        <select id="runSelect" multiple></select>
      </div>
      <div class="control">
        <label for="tableSelect">Metric family</label>
        <select id="tableSelect"></select>
      </div>
      <div class="control">
        <label for="metricSelect">Metric</label>
        <select id="metricSelect"></select>
      </div>
      <div class="control" id="qControl">
        <label for="qSelect">H quantile</label>
        <select id="qSelect"></select>
      </div>
      <div class="control">
        <label for="betaSelect">Beta cells</label>
        <select id="betaSelect" multiple></select>
        <div class="buttons">
          <button type="button" data-beta="all">All</button>
          <button type="button" data-beta="low">Low</button>
          <button type="button" data-beta="high">High</button>
        </div>
      </div>
      <div class="control row">
        <div>
          <label for="rMin">Radius min</label>
          <input id="rMin" type="number" step="0.01" />
        </div>
        <div>
          <label for="rMax">Radius max</label>
          <input id="rMax" type="number" step="0.01" />
        </div>
      </div>
      <div class="control row">
        <div>
          <label for="plotMode">View</label>
          <select id="plotMode">
            <option value="line">Lines</option>
            <option value="heatmap">Heatmap</option>
          </select>
        </div>
        <div>
          <label for="xScale">X scale</label>
          <select id="xScale">
            <option value="linear">Linear</option>
            <option value="log">Log</option>
          </select>
        </div>
      </div>
      <div class="control">
        <label for="yTransform">Y/color transform</label>
        <select id="yTransform">
          <option value="linear">Linear</option>
          <option value="signedLog">Signed log10(1 + abs(x))</option>
        </select>
      </div>
      <button class="primary" type="button" id="renderBtn">Render</button>
      <div class="note">
        Heatmap uses the first selected run. Line view can overlay both runs.
        Signed-log transform is useful for mixed-sign energy and derivative
        values; hover values still show the original metric.
      </div>
    </aside>
    <section class="chart-panel">
      <div id="summary"></div>
      <div id="plot"></div>
    </section>
  </main>
  <script id="payload" type="application/json">{payload}</script>
  <script>
    const PAYLOAD = JSON.parse(document.getElementById("payload").textContent);
    const runs = PAYLOAD.runs;
    const metricLabels = PAYLOAD.metricLabels;
    const tableLabels = {{
      absolute: "Absolute phi",
      delta: "Delta phi",
      derivative: "Radial derivative",
      hq: "H threshold"
    }};
    const colors = ["#0f766e", "#7c3aed", "#b45309", "#2563eb", "#be123c", "#4d7c0f", "#4338ca", "#a21caf"];

    const el = {{
      runSelect: document.getElementById("runSelect"),
      tableSelect: document.getElementById("tableSelect"),
      metricSelect: document.getElementById("metricSelect"),
      qControl: document.getElementById("qControl"),
      qSelect: document.getElementById("qSelect"),
      betaSelect: document.getElementById("betaSelect"),
      rMin: document.getElementById("rMin"),
      rMax: document.getElementById("rMax"),
      plotMode: document.getElementById("plotMode"),
      xScale: document.getElementById("xScale"),
      yTransform: document.getElementById("yTransform"),
      renderBtn: document.getElementById("renderBtn"),
      summary: document.getElementById("summary"),
      plot: document.getElementById("plot")
    }};

    function option(value, text, selected = false) {{
      const node = document.createElement("option");
      node.value = value;
      node.textContent = text;
      node.selected = selected;
      return node;
    }}

    function selectedValues(select) {{
      return Array.from(select.selectedOptions).map((item) => item.value);
    }}

    function fmtBeta(beta) {{
      return Number(beta).toFixed(2);
    }}

    function transformValue(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
      const numeric = Number(value);
      if (el.yTransform.value === "signedLog") {{
        return Math.sign(numeric) * Math.log10(1 + Math.abs(numeric));
      }}
      return numeric;
    }}

    function setup() {{
      runs.forEach((run, index) => {{
        el.runSelect.appendChild(option(run.id, run.label, index === 0));
      }});
      Object.keys(tableLabels).forEach((key) => {{
        el.tableSelect.appendChild(option(key, tableLabels[key], key === "absolute"));
      }});
      updateMetricOptions();
      updateBetaOptions();
      updateRadiusInputs();
      updateQOptions();
      render();
    }}

    function getSelectedRuns() {{
      const ids = selectedValues(el.runSelect);
      return runs.filter((run) => ids.includes(run.id));
    }}

    function activeTable() {{
      return el.tableSelect.value;
    }}

    function updateMetricOptions() {{
      const table = activeTable();
      const selectedRuns = getSelectedRuns();
      const available = new Set();
      selectedRuns.forEach((run) => {{
        (run.metrics[table] || []).forEach(([key]) => available.add(key));
      }});
      el.metricSelect.innerHTML = "";
      const candidates = metricLabels[table].filter(([key]) => available.has(key));
      candidates.forEach(([key, label], index) => {{
        el.metricSelect.appendChild(option(key, label, index === 0));
      }});
      el.qControl.style.display = table === "hq" ? "block" : "none";
    }}

    function updateBetaOptions(keepSelection = false) {{
      const old = new Set(selectedValues(el.betaSelect));
      const betas = new Set();
      getSelectedRuns().forEach((run) => run.betas.forEach((beta) => betas.add(fmtBeta(beta))));
      const sorted = Array.from(betas).sort((a, b) => Number(a) - Number(b));
      el.betaSelect.innerHTML = "";
      sorted.forEach((beta) => {{
        const selected = keepSelection ? old.has(beta) : true;
        el.betaSelect.appendChild(option(beta, beta, selected));
      }});
    }}

    function updateRadiusInputs() {{
      const values = [];
      getSelectedRuns().forEach((run) => run.radii.forEach((radius) => values.push(radius)));
      const min = Math.min(...values);
      const max = Math.max(...values);
      if (Number.isFinite(min)) el.rMin.value = min.toFixed(2);
      if (Number.isFinite(max)) el.rMax.value = max.toFixed(2);
    }}

    function updateQOptions() {{
      const qs = new Set();
      getSelectedRuns().forEach((run) => run.qs.forEach((q) => qs.add(Number(q).toFixed(2))));
      el.qSelect.innerHTML = "";
      Array.from(qs).sort((a, b) => Number(a) - Number(b)).forEach((q, index) => {{
        el.qSelect.appendChild(option(q, q, index === 0));
      }});
    }}

    function filteredRows(run) {{
      const table = activeTable();
      const metric = el.metricSelect.value;
      const betas = new Set(selectedValues(el.betaSelect));
      const rMin = Number(el.rMin.value);
      const rMax = Number(el.rMax.value);
      const q = Number(el.qSelect.value);
      return (run.tables[table] || []).filter((row) => {{
        if (!betas.has(fmtBeta(row.beta))) return false;
        if (row.radius < rMin || row.radius > rMax) return false;
        if (table === "hq" && Number(row.q).toFixed(2) !== q.toFixed(2)) return false;
        return row[metric] !== null && row[metric] !== undefined;
      }});
    }}

    function renderSummary(rowsByRun) {{
      const table = activeTable();
      const metric = el.metricSelect.value;
      const allRows = rowsByRun.flatMap((item) => item.rows);
      const values = allRows.map((row) => Number(row[metric])).filter(Number.isFinite);
      const betaCount = new Set(allRows.map((row) => fmtBeta(row.beta))).size;
      const radiusCount = new Set(allRows.map((row) => Number(row.radius).toFixed(4))).size;
      const min = values.length ? Math.min(...values) : null;
      const max = values.length ? Math.max(...values) : null;
      const stats = [
        ["Runs", rowsByRun.length],
        ["Beta cells", betaCount],
        ["Radii", radiusCount],
        ["Metric range", values.length ? `${min.toPrecision(5)} to ${max.toPrecision(5)}` : "n/a"],
      ];
      el.summary.innerHTML = stats.map(([key, value]) => (
        `<div class="stat"><div class="k">${key}</div><div class="v">${value}</div></div>`
      )).join("");
    }}

    function renderLine(rowsByRun) {{
      const table = activeTable();
      const metric = el.metricSelect.value;
      const traces = [];
      rowsByRun.forEach((item, runIndex) => {{
        const byBeta = new Map();
        item.rows.forEach((row) => {{
          const beta = fmtBeta(row.beta);
          if (!byBeta.has(beta)) byBeta.set(beta, []);
          byBeta.get(beta).push(row);
        }});
        Array.from(byBeta.entries()).sort((a, b) => Number(a[0]) - Number(b[0])).forEach(([beta, rows], betaIndex) => {{
          rows.sort((a, b) => a.radius - b.radius);
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: `${item.run.label} beta ${beta}`,
            x: rows.map((row) => row.radius),
            y: rows.map((row) => transformValue(row[metric])),
            customdata: rows.map((row) => [row[metric], row.beta, row.radius, item.run.id]),
            hovertemplate: "run=%{customdata[3]}<br>beta=%{customdata[1]:.2f}<br>radius=%{customdata[2]:.3f}<br>value=%{customdata[0]:.6g}<extra></extra>",
            line: {{
              width: 1.7,
              color: colors[(betaIndex + runIndex * 3) % colors.length],
              dash: runIndex === 0 ? "solid" : "dot"
            }}
          }});
        }});
      }});
      Plotly.react(el.plot, traces, layout(`${tableLabels[table]}: ${metric}`));
    }}

    function renderHeatmap(rowsByRun) {{
      const item = rowsByRun[0];
      if (!item) return;
      const metric = el.metricSelect.value;
      const table = activeTable();
      const betas = Array.from(new Set(item.rows.map((row) => fmtBeta(row.beta)))).sort((a, b) => Number(a) - Number(b));
      const radii = Array.from(new Set(item.rows.map((row) => Number(row.radius).toFixed(4)))).sort((a, b) => Number(a) - Number(b));
      const rowMap = new Map(item.rows.map((row) => [`${fmtBeta(row.beta)}|${Number(row.radius).toFixed(4)}`, row]));
      const z = betas.map((beta) => radii.map((radius) => {{
        const row = rowMap.get(`${beta}|${radius}`);
        return row ? transformValue(row[metric]) : null;
      }}));
      const raw = betas.map((beta) => radii.map((radius) => {{
        const row = rowMap.get(`${beta}|${radius}`);
        return row ? row[metric] : null;
      }}));
      const trace = {{
        type: "heatmap",
        x: radii.map(Number),
        y: betas,
        z,
        customdata: raw,
        colorscale: "Viridis",
        colorbar: {{ title: metric }},
        hovertemplate: "beta=%{y}<br>radius=%{x:.3f}<br>value=%{customdata:.6g}<extra></extra>"
      }};
      Plotly.react(el.plot, [trace], layout(`${item.run.label}: ${tableLabels[table]} heatmap ${metric}`));
    }}

    function layout(title) {{
      return {{
        title: {{ text: title, font: {{ size: 15 }} }},
        margin: {{ l: 70, r: 25, t: 55, b: 60 }},
        xaxis: {{
          title: "radius",
          type: el.xScale.value,
          zeroline: false,
          gridcolor: "#e8ebf0"
        }},
        yaxis: {{
          title: el.yTransform.value === "signedLog" ? "signed-log transformed value" : "value",
          zeroline: false,
          gridcolor: "#e8ebf0"
        }},
        legend: {{ orientation: "h", y: -0.18 }},
        hovermode: "closest",
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff"
      }};
    }}

    function render() {{
      const rowsByRun = getSelectedRuns().map((run) => ({{ run, rows: filteredRows(run) }}));
      renderSummary(rowsByRun);
      if (el.plotMode.value === "heatmap") {{
        renderHeatmap(rowsByRun);
      }} else {{
        renderLine(rowsByRun);
      }}
    }}

    el.runSelect.addEventListener("change", () => {{
      updateMetricOptions();
      updateBetaOptions(true);
      updateRadiusInputs();
      updateQOptions();
      render();
    }});
    el.tableSelect.addEventListener("change", () => {{
      updateMetricOptions();
      updateQOptions();
      render();
    }});
    el.metricSelect.addEventListener("change", render);
    el.qSelect.addEventListener("change", render);
    el.betaSelect.addEventListener("change", render);
    el.rMin.addEventListener("change", render);
    el.rMax.addEventListener("change", render);
    el.plotMode.addEventListener("change", render);
    el.xScale.addEventListener("change", render);
    el.yTransform.addEventListener("change", render);
    el.renderBtn.addEventListener("click", render);
    document.querySelectorAll("[data-beta]").forEach((button) => {{
      button.addEventListener("click", () => {{
        const mode = button.dataset.beta;
        Array.from(el.betaSelect.options).forEach((opt) => {{
          const beta = Number(opt.value);
          opt.selected = mode === "all" || (mode === "low" && beta <= 0.19) || (mode === "high" && beta >= 0.25);
        }});
        render();
      }});
    }});

    setup();
  </script>
</body>
</html>
"""


def write_dashboard(path: Path, runs: list[dict], plotly_src: str) -> None:
    payload = {
        "runs": runs,
        "metricLabels": METRICS,
    }
    html = (
        HTML_TEMPLATE.replace("{{", "{")
        .replace("}}", "}")
        .replace("{plotly_src}", plotly_src)
        .replace("{payload}", json.dumps(payload, separators=(",", ":")))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def ensure_plotly_asset(repo_root: Path) -> None:
    src = (
        repo_root
        / "02_dnn/07_proxy_3D_landscape_for_visualize/figures/"
        / "proxy_landscape_v9_reviewed_standalone/vendor/plotly-2.30.0.min.js"
    )
    dst = (
        repo_root
        / "02_dnn/05_proxy_local_entropy/figures/_assets/plotly-2.30.0.min.js"
    )
    if not src.exists():
        raise FileNotFoundError(f"Plotly vendor bundle not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build(repo_root: Path) -> None:
    ensure_plotly_asset(repo_root)
    loaded = [load_run(repo_root, run) for run in RUNS]
    figure_root = repo_root / "02_dnn/05_proxy_local_entropy/figures"
    write_dashboard(
        figure_root / "local_entropy_dashboard.html",
        loaded,
        "_assets/plotly-2.30.0.min.js",
    )
    for run in loaded:
        write_dashboard(
            figure_root / run["id"] / run["range"] / "local_entropy_dashboard.html",
            [run],
            "../../_assets/plotly-2.30.0.min.js",
        )
    readme = """# Proxy Local Entropy Figures

The active analysis figure is the interactive dashboard built from the retained
summary tables. Static PNG/CSV/report clutter from older figure passes is not
part of the active 05-stage figure contract.

- `local_entropy_dashboard.html`: combined 30-dataset and 60-dataset dashboard.
- `{run_30}/d_0.01_to_2.50_dense/local_entropy_dashboard.html`: 30-dataset view.
- `{run_60}/d_0.01_to_2.50_dense/local_entropy_dashboard.html`: 60-dataset view.
- `_assets/plotly-2.30.0.min.js`: local Plotly bundle used by the dashboards.

Use the dashboard controls for run selection, metric family, metric, beta
selection, radius window, linear/log radius scale, signed-log value transform,
and line/heatmap view.

Rebuild from the repository root with:

```powershell
python 02_dnn/05_proxy_local_entropy/src/build_interactive_dashboard.py --repo-root .
```
""".format(
        run_30=RUNS[0]["id"],
        run_60=RUNS[1]["id"],
    )
    (figure_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    build(args.repo_root.resolve())


if __name__ == "__main__":
    main()
