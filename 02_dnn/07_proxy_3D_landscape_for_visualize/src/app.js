const PATHS = {
  payload: "../../raw_outputs/proxy_landscape_v9_reviewed_fit/plotly_payload_v9.json",
  fitSummary: "../../raw_outputs/proxy_landscape_v9_reviewed_fit/proxy_fit_summary_v9.csv",
  targetAggregate:
    "../../raw_outputs/reference_atlas_target_measures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/geometry_summary_beta_aggregate.csv",
  phiDescriptor:
    "../../raw_outputs/reference_atlas_target_measures/18_beta_cell_30_dataset_30_reference/d_0.01_to_2.50_dense/phi_E_descriptor_by_beta.csv",
};

const state = {
  payload: null,
  fitRows: [],
  targetRows: [],
  phiRows: [],
  targetByBeta: new Map(),
  phiByBeta: new Map(),
  index: 0,
  camera: null,
  timer: null,
  surfaceReady: false,
  lossRange: null,
  xyRange: null,
};

const el = {
  betaLabel: document.getElementById("betaLabel"),
  betaSlider: document.getElementById("betaSlider"),
  betaRail: document.getElementById("betaRail"),
  playBtn: document.getElementById("playBtn"),
  isoBtn: document.getElementById("isoBtn"),
  topBtn: document.getElementById("topBtn"),
  sideBtn: document.getElementById("sideBtn"),
  surfacePlot: document.getElementById("surfacePlot"),
  metricGrid: document.getElementById("metricGrid"),
  fitStatus: document.getElementById("fitStatus"),
  residualPlot: document.getElementById("residualPlot"),
  descriptorPlot: document.getElementById("descriptorPlot"),
};

async function loadText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.text();
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json();
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines
    .filter(Boolean)
    .map((line) => {
      const values = line.split(",");
      return Object.fromEntries(
        headers.map((header, index) => [header, coerce(values[index])])
      );
    });
}

function coerce(value) {
  if (value === undefined) return "";
  const trimmed = value.trim();
  if (trimmed === "") return "";
  const number = Number(trimmed);
  return Number.isFinite(number) ? number : trimmed;
}

function betaKey(beta) {
  return Number(beta).toFixed(2);
}

function fmt(value, digits = 3) {
  if (!Number.isFinite(Number(value))) return "--";
  const number = Number(value);
  if (Math.abs(number) >= 100) return number.toFixed(1);
  if (Math.abs(number) >= 10) return number.toFixed(2);
  return number.toFixed(digits);
}

function signedFmt(value, digits = 3) {
  if (!Number.isFinite(Number(value))) return "--";
  return `${value >= 0 ? "+" : ""}${fmt(value, digits)}`;
}

function metricResiduals(fitRow) {
  const target = state.targetByBeta.get(betaKey(fitRow.beta)) || {};
  const specs = [
    {
      id: "S_ref",
      label: "S",
      target: fitRow.S_target,
      proxy: fitRow.S_proxy,
      sem: target.S_ref_sem,
    },
    {
      id: "H_CE",
      label: "H_CE",
      target: fitRow.H_CE_target,
      proxy: fitRow.H_CE_proxy,
      sem: target.H_CE_sem,
    },
    {
      id: "B_med",
      label: "B_med",
      target: fitRow.B_med_target,
      proxy: fitRow.B_med_proxy,
      sem: target.B_CE_median_sem,
    },
    {
      id: "B_q90",
      label: "B_q90",
      target: fitRow.B_q90_target,
      proxy: fitRow.B_q90_proxy,
      sem: target.B_CE_q90_sem,
    },
  ];

  return specs.map((spec) => {
    const error = Number(spec.proxy) - Number(spec.target);
    const sem = Number(spec.sem);
    const z = Number.isFinite(sem) && sem > 0 ? error / sem : 0;
    return { ...spec, error, sem, z, absZ: Math.abs(z) };
  });
}

function statusForResiduals(residuals) {
  const largest = [...residuals].sort((a, b) => b.absZ - a.absZ)[0];
  if (!largest || largest.absZ <= 1) {
    return {
      className: "ok",
      label: "Within target SEM",
      text: "All displayed fit residuals are within one target SEM for this beta.",
      largest,
    };
  }
  if (largest.absZ <= 2) {
    return {
      className: "watch",
      label: "Watch residual",
      text: `${largest.label} is ${signedFmt(largest.z, 2)} SEM from the copied target measure.`,
      largest,
    };
  }
  return {
    className: "risk",
    label: "Conservative read",
    text: `${largest.label} is ${signedFmt(largest.z, 2)} SEM from the copied target measure; tail-barrier shape should be read conservatively here.`,
    largest,
  };
}

function valueAtNearestGrid(surface, xValue, yValue) {
  const xIndex = nearestIndex(state.payload.x, xValue);
  const yIndex = nearestIndex(state.payload.y, yValue);
  return surface.z[yIndex][xIndex];
}

function computeSurfaceDomains(payload) {
  const xRange = widenedRange(payload.x, 0.18);
  const yRange = widenedRange(payload.y, 0.18);
  let zMin = Infinity;
  let zMax = -Infinity;

  payload.surfaces.forEach((surface) => {
    surface.z.forEach((row) => {
      row.forEach((value) => {
        const z = Number(value);
        if (z < zMin) zMin = z;
        if (z > zMax) zMax = z;
      });
    });
  });

  return {
    lossRange: [zMin, zMax],
    xyRange: { x: xRange, y: yRange },
  };
}

function widenedRange(values, fraction) {
  const numeric = values.map(Number);
  const min = Math.min(...numeric);
  const max = Math.max(...numeric);
  const margin = (max - min) * fraction * 0.5;
  return [roundForAxis(min - margin), roundForAxis(max + margin)];
}

function roundForAxis(value) {
  return Math.sign(value) * Math.ceil(Math.abs(value) * 10) / 10;
}

function nearestIndex(values, target) {
  let bestIndex = 0;
  let bestDistance = Infinity;
  values.forEach((value, index) => {
    const distance = Math.abs(Number(value) - Number(target));
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function updateSurface(index) {
  const surface = state.payload.surfaces[index];
  const fitRow = state.fitRows[index] || state.payload.fit[index];
  const beta = betaKey(surface.beta);
  const wells = surface.wells || [];
  const wellX = wells.map((point) => point[0]);
  const wellY = wells.map((point) => point[1]);
  const wellZ = wells.map((point) => valueAtNearestGrid(surface, point[0], point[1]));
  const centerZ = valueAtNearestGrid(surface, 0, 0);

  const traces = [
    {
      type: "surface",
      x: state.payload.x,
      y: state.payload.y,
      z: surface.z,
      cmin: state.lossRange[0],
      cmax: state.lossRange[1],
      colorscale: "Portland",
      colorbar: {
        title: "proxy loss",
        len: 0.74,
        thickness: 14,
      },
      contours: {
        z: {
          show: true,
          usecolormap: true,
          highlightcolor: "#20231f",
          project: { z: true },
        },
      },
      hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>L=%{z:.2f}<extra></extra>",
    },
    {
      type: "scatter3d",
      mode: "markers",
      name: "reference wells",
      x: wellX,
      y: wellY,
      z: wellZ,
      marker: {
        color: "#ffffff",
        line: { color: "#20231f", width: 1 },
        size: 2.25,
        symbol: "circle",
      },
      hovertemplate: "reference well<br>x=%{x:.2f}<br>y=%{y:.2f}<br>L=%{z:.2f}<extra></extra>",
    },
    {
      type: "scatter3d",
      mode: "markers",
      name: "mean center",
      x: [0],
      y: [0],
      z: [centerZ],
      marker: {
        color: "#b23b3b",
        line: { color: "#ffffff", width: 1 },
        size: 4,
        symbol: "diamond",
      },
      hovertemplate: "mean center<br>L=%{z:.2f}<extra></extra>",
    },
  ];

  const layout = {
    margin: { l: 0, r: 0, t: 34, b: 0 },
    paper_bgcolor: "#fbfcfa",
    plot_bgcolor: "#fbfcfa",
    showlegend: true,
    title: {
      text: `beta ${beta} | Umax ${fmt(fitRow.Umax_total_energy || surface.Umax, 2)} | ridge ${fmt(fitRow.ridge_coef || surface.a, 2)} | center ${fmt(fitRow.center_coef || surface.c, 2)}`,
      font: { size: 14, color: "#20231f" },
      x: 0.02,
    },
    legend: {
      bgcolor: "rgba(255,255,255,0.82)",
      bordercolor: "#d6dbd1",
      borderwidth: 1,
      x: 0.02,
      y: 0.98,
    },
    scene: {
      camera: state.camera || {
        eye: { x: 1.55, y: 1.55, z: 0.92 },
        up: { x: 0, y: 0, z: 1 },
      },
      xaxis: axis3d("latent x", { range: state.xyRange.x }),
      yaxis: axis3d("latent y", { range: state.xyRange.y }),
      zaxis: axis3d("proxy loss", { range: state.lossRange }),
      aspectmode: "cube",
    },
    uirevision: "preserve-camera",
  };

  const config = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  Plotly.react(el.surfacePlot, traces, layout, config);
  if (!state.surfaceReady) {
    el.surfacePlot.on("plotly_relayout", (event) => {
      if (event["scene.camera"]) {
        state.camera = event["scene.camera"];
      }
    });
    state.surfaceReady = true;
  }
}

function axis3d(title, options = {}) {
  const axis = {
    title,
    backgroundcolor: "#f8faf7",
    gridcolor: "#d6dbd1",
    showbackground: true,
    zerolinecolor: "#aeb8a9",
  };
  if (options.range) {
    axis.autorange = false;
    axis.range = options.range;
  }
  return axis;
}

function updateDiagnostics(index) {
  const fitRow = state.fitRows[index];
  const beta = betaKey(fitRow.beta);
  const target = state.targetByBeta.get(beta) || {};
  const phi = state.phiByBeta.get(beta) || {};
  const residuals = metricResiduals(fitRow);
  const status = statusForResiduals(residuals);

  el.betaLabel.textContent = beta;
  el.fitStatus.className = `fit-status ${status.className}`;
  el.fitStatus.textContent = `${status.label}: ${status.text}`;

  const cards = [
    metricCard("S target/proxy", fitRow.S_target, fitRow.S_proxy, target.S_ref_sem),
    metricCard("Q target/proxy", fitRow.Q_target, fitRow.Q_proxy_latent, target.Q_ref_sem),
    metricCard("H_CE target/proxy", fitRow.H_CE_target, fitRow.H_CE_proxy, target.H_CE_sem),
    metricCard("B_med target/proxy", fitRow.B_med_target, fitRow.B_med_proxy, target.B_CE_median_sem),
    metricCard("B_q90 target/proxy", fitRow.B_q90_target, fitRow.B_q90_proxy, target.B_CE_q90_sem),
    {
      label: "phi_E(d=2.5)",
      value: fmt(phi.phi_E ?? fitRow.phi_E_2p5_target, 5),
      sub: `ref_count ${phi.ref_count ?? "--"}, dataset_count ${phi.dataset_count ?? "--"}`,
    },
  ];

  el.metricGrid.innerHTML = cards
    .map(
      (card) => `
        <div class="metric-card">
          <span class="label">${card.label}</span>
          <span class="value">${card.value}</span>
          <span class="sub">${card.sub}</span>
        </div>
      `
    )
    .join("");

  updateResidualPlot(residuals);
  updateBetaRail();
}

function metricCard(label, target, proxy, sem) {
  const error = Number(proxy) - Number(target);
  const semText = Number.isFinite(Number(sem)) ? `target SEM ${fmt(sem, 4)}` : "target SEM --";
  return {
    label,
    value: `${fmt(target, 4)} / ${fmt(proxy, 4)}`,
    sub: `residual ${signedFmt(error, 4)} | ${semText}`,
  };
}

function updateResidualPlot(residuals) {
  const colors = residuals.map((item) =>
    item.absZ <= 1 ? "#21745a" : item.absZ <= 2 ? "#a76b00" : "#b23b3b"
  );

  const trace = {
    type: "bar",
    x: residuals.map((item) => item.label),
    y: residuals.map((item) => item.z),
    marker: { color: colors },
    text: residuals.map((item) => signedFmt(item.z, 2)),
    textposition: "outside",
    hovertemplate:
      "%{x}<br>residual/SEM=%{y:.2f}<extra></extra>",
  };

  const layout = {
    margin: { l: 48, r: 18, t: 20, b: 42 },
    paper_bgcolor: "#fbfcfa",
    plot_bgcolor: "#fbfcfa",
    yaxis: {
      title: "residual / target SEM",
      zeroline: true,
      zerolinecolor: "#20231f",
      gridcolor: "#d6dbd1",
    },
    xaxis: { tickfont: { size: 12 } },
    shapes: [
      thresholdLine(2),
      thresholdLine(-2),
    ],
  };

  Plotly.react(el.residualPlot, [trace], layout, plotConfig());
}

function thresholdLine(y) {
  return {
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    y0: y,
    y1: y,
    line: { color: "#b23b3b", width: 1, dash: "dot" },
  };
}

function updateDescriptorPlot() {
  const betaValues = state.targetRows.map((row) => row.beta);
  const series = [
    { key: "S_ref_mean", name: "S_ref", color: "#21745a" },
    { key: "H_CE_mean", name: "H_CE", color: "#7c5d00" },
    { key: "B_CE_median_mean", name: "B_med", color: "#b23b3b" },
    { key: "B_CE_q90_mean", name: "B_q90", color: "#734b96" },
    { key: "phi_E_dstar", name: "phi_E(2.5)", color: "#2f6f8f" },
  ];

  const traces = series.map((spec) => ({
    type: "scatter",
    mode: "lines+markers",
    name: spec.name,
    x: betaValues,
    y: normalize(state.targetRows.map((row) => Number(row[spec.key]))),
    line: { color: spec.color, width: 2.5 },
    marker: { size: 6, color: spec.color },
    customdata: state.targetRows.map((row) => row[spec.key]),
    hovertemplate: `${spec.name}<br>beta=%{x:.2f}<br>normalized=%{y:.3f}<br>raw=%{customdata:.4f}<extra></extra>`,
  }));

  const currentBeta = Number(state.fitRows[state.index]?.beta || betaValues[0]);
  const layout = {
    margin: { l: 48, r: 22, t: 18, b: 42 },
    paper_bgcolor: "#fbfcfa",
    plot_bgcolor: "#fbfcfa",
    legend: { orientation: "h", y: 1.08 },
    xaxis: {
      title: "beta",
      gridcolor: "#d6dbd1",
      zeroline: false,
    },
    yaxis: {
      title: "normalized target measure",
      range: [-0.05, 1.05],
      gridcolor: "#d6dbd1",
    },
    shapes: [
      {
        type: "line",
        x0: currentBeta,
        x1: currentBeta,
        y0: 0,
        y1: 1,
        xref: "x",
        yref: "paper",
        line: { color: "#20231f", width: 2 },
      },
    ],
  };

  Plotly.react(el.descriptorPlot, traces, layout, plotConfig());
}

function normalize(values) {
  const finite = values.filter(Number.isFinite);
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (max === min) return values.map(() => 0.5);
  return values.map((value) => (Number(value) - min) / (max - min));
}

function updateBetaRail() {
  const buttons = [...el.betaRail.querySelectorAll(".beta-dot")];
  buttons.forEach((button, index) => {
    button.classList.toggle("active", index === state.index);
  });
}

function buildBetaRail() {
  el.betaRail.innerHTML = state.fitRows
    .map((row, index) => {
      const status = statusForResiduals(metricResiduals(row));
      return `<button class="beta-dot ${status.className}" data-index="${index}" title="beta ${betaKey(row.beta)}">${row.beta.toFixed(2)}</button>`;
    })
    .join("");

  [...el.betaRail.querySelectorAll(".beta-dot")].forEach((button) => {
    button.addEventListener("click", () => {
      setIndex(Number(button.dataset.index));
    });
  });
}

function setIndex(index) {
  state.index = Math.max(0, Math.min(index, state.fitRows.length - 1));
  el.betaSlider.value = String(state.index);
  updateSurface(state.index);
  updateDiagnostics(state.index);
  updateDescriptorPlot();
}

function plotConfig() {
  return {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };
}

function setCamera(camera) {
  state.camera = camera;
  Plotly.relayout(el.surfacePlot, { "scene.camera": camera });
}

function wireControls() {
  el.betaSlider.max = String(state.fitRows.length - 1);
  el.betaSlider.addEventListener("input", () => setIndex(Number(el.betaSlider.value)));

  el.playBtn.addEventListener("click", () => {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
      el.playBtn.querySelector("span").textContent = "Play";
      el.playBtn.querySelector("svg")?.setAttribute("data-lucide", "play");
      if (window.lucide) window.lucide.createIcons();
      return;
    }

    el.playBtn.querySelector("span").textContent = "Pause";
    el.playBtn.querySelector("svg")?.setAttribute("data-lucide", "pause");
    if (window.lucide) window.lucide.createIcons();
    state.timer = setInterval(() => {
      setIndex((state.index + 1) % state.fitRows.length);
    }, 900);
  });

  el.isoBtn.addEventListener("click", () =>
    setCamera({ eye: { x: 1.55, y: 1.55, z: 0.92 }, up: { x: 0, y: 0, z: 1 } })
  );
  el.topBtn.addEventListener("click", () =>
    setCamera({ eye: { x: 0, y: 0, z: 2.25 }, up: { x: 0, y: 1, z: 0 } })
  );
  el.sideBtn.addEventListener("click", () =>
    setCamera({ eye: { x: 2.35, y: 0.05, z: 0.35 }, up: { x: 0, y: 0, z: 1 } })
  );
}

async function init() {
  try {
    const [payload, fitCsv, targetCsv, phiCsv] = await Promise.all([
      loadJson(PATHS.payload),
      loadText(PATHS.fitSummary),
      loadText(PATHS.targetAggregate),
      loadText(PATHS.phiDescriptor),
    ]);

    state.payload = payload;
    const domains = computeSurfaceDomains(payload);
    state.lossRange = domains.lossRange;
    state.xyRange = domains.xyRange;
    state.fitRows = parseCsv(fitCsv);
    state.targetRows = parseCsv(targetCsv);
    state.phiRows = parseCsv(phiCsv);
    state.targetByBeta = new Map(state.targetRows.map((row) => [betaKey(row.beta), row]));
    state.phiByBeta = new Map(state.phiRows.map((row) => [betaKey(row.beta), row]));

    buildBetaRail();
    wireControls();
    setIndex(0);
    if (window.lucide) window.lucide.createIcons();
  } catch (error) {
    document.body.insertAdjacentHTML(
      "afterbegin",
      `<div class="error-box">Failed to load proxy landscape data: ${error.message}</div>`
    );
    throw error;
  }
}

init();
