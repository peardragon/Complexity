const fs = require("fs");
const path = require("path");
const https = require("https");

const STAGE_ROOT = path.resolve(__dirname, "..");
const PAYLOAD_PATH = path.join(
  STAGE_ROOT,
  "raw_outputs",
  "proxy_landscape_v9_reviewed_fit",
  "plotly_payload_v9.json"
);
const OUT_DIR = path.join(
  STAGE_ROOT,
  "figures",
  "proxy_landscape_v9_reviewed_standalone"
);
const VENDOR_DIR = path.join(OUT_DIR, "vendor");
const PLOTLY_PATH = path.join(VENDOR_DIR, "plotly-2.30.0.min.js");
const OUT_HTML = path.join(OUT_DIR, "proxy_3d_landscape_only.html");
const PLOTLY_URL = "https://cdn.plot.ly/plotly-2.30.0.min.js";

function download(url, destination) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destination);
    https
      .get(url, (response) => {
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
          file.close();
          fs.unlinkSync(destination);
          download(response.headers.location, destination).then(resolve, reject);
          return;
        }
        if (response.statusCode !== 200) {
          file.close();
          fs.unlinkSync(destination);
          reject(new Error(`Download failed with HTTP ${response.statusCode}: ${url}`));
          return;
        }
        response.pipe(file);
        file.on("finish", () => {
          file.close(resolve);
        });
      })
      .on("error", (error) => {
        file.close();
        if (fs.existsSync(destination)) fs.unlinkSync(destination);
        reject(error);
      });
  });
}

function computeSurfaceDomains(payload) {
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
    xyRange: {
      x: widenedRange(payload.x, 0.18),
      y: widenedRange(payload.y, 0.18),
    },
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

function escapeScriptContent(text) {
  return text.replace(/<\/script/gi, "<\\/script").replace(/<!--/g, "<\\!--");
}

function html({ payloadText, plotlyText, domains }) {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Proxy 3D Landscape Sweep</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f5f1;
      --panel: #ffffff;
      --panel-strong: #f8faf7;
      --ink: #20231f;
      --muted: #656b61;
      --line: #d6dbd1;
      --green: #21745a;
      --green-weak: #dceee5;
      --charcoal: #343832;
      --plot-bg: #fbfcfa;
      --shadow: 0 12px 32px rgba(32, 35, 31, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
      padding: 22px;
      gap: 14px;
    }
    header {
      align-items: center;
      display: flex;
      gap: 16px;
      justify-content: space-between;
    }
    h1 {
      font-size: 23px;
      line-height: 1.15;
      margin: 0;
    }
    .beta-readout {
      align-items: flex-end;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      min-width: 96px;
      padding: 10px 12px;
    }
    .beta-readout span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .beta-readout strong {
      font-size: 25px;
      line-height: 1.1;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-height: 0;
      padding: 16px;
    }
    .control-row {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
    }
    .slider-label {
      align-items: center;
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 8px;
      display: flex;
      gap: 12px;
      min-height: 42px;
      min-width: min(520px, 100%);
      padding: 8px 12px;
    }
    .slider-label span {
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
      white-space: nowrap;
    }
    input[type="range"] {
      accent-color: var(--green);
      flex: 1 1 280px;
      min-width: 180px;
    }
    button {
      align-items: center;
      background: var(--charcoal);
      border: 1px solid var(--charcoal);
      border-radius: 8px;
      color: #fff;
      cursor: pointer;
      display: inline-flex;
      font: inherit;
      font-size: 14px;
      font-weight: 800;
      gap: 8px;
      min-height: 42px;
      padding: 8px 12px;
    }
    button:hover { background: #1d201c; }
    .beta-rail {
      display: grid;
      gap: 4px;
      grid-template-columns: repeat(18, minmax(30px, 1fr));
      margin: 0 0 12px;
    }
    .beta-dot {
      align-items: center;
      background: var(--green-weak);
      border: 1px solid #a9d0bd;
      border-radius: 6px;
      color: var(--ink);
      cursor: pointer;
      display: flex;
      font-size: 11px;
      font-weight: 800;
      height: 30px;
      justify-content: center;
      padding: 0;
    }
    .beta-dot.active {
      outline: 2px solid var(--charcoal);
      outline-offset: 1px;
    }
    #plot {
      background: var(--plot-bg);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 520px;
      overflow: hidden;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-left: auto;
    }
    @media (max-width: 760px) {
      .shell { padding: 10px; }
      header { align-items: flex-start; flex-direction: column; }
      .beta-rail { grid-template-columns: repeat(6, minmax(34px, 1fr)); }
      .panel { padding: 10px; }
      #plot { min-height: 520px; }
    }
  </style>
  <script>${escapeScriptContent(plotlyText)}</script>
</head>
<body>
  <div class="shell">
    <header>
      <h1>Proxy 3D Landscape Sweep</h1>
      <div class="beta-readout">
        <span>beta</span>
        <strong id="betaLabel">--</strong>
      </div>
    </header>
    <main class="panel">
      <div class="control-row">
        <label class="slider-label" for="betaSlider">
          <span>beta sweep</span>
          <input id="betaSlider" type="range" min="0" max="17" value="0" step="1" />
        </label>
        <button id="playBtn" type="button">Play</button>
        <button id="isoBtn" type="button">Isometric</button>
        <button id="topBtn" type="button">Top</button>
        <button id="sideBtn" type="button">Side</button>
        <span class="meta">fixed proxy-loss scale: ${domains.lossRange[0].toFixed(3)} to ${domains.lossRange[1].toFixed(3)}</span>
      </div>
      <div id="betaRail" class="beta-rail" aria-label="beta sweep shortcuts"></div>
      <div id="plot" role="img" aria-label="interactive proxy 3D landscape"></div>
    </main>
  </div>
  <script id="payload-json" type="application/json">${escapeScriptContent(payloadText)}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload-json").textContent);
    const lossRange = ${JSON.stringify(domains.lossRange)};
    const xyRange = ${JSON.stringify(domains.xyRange)};
    let index = 0;
    let camera = null;
    let timer = null;
    let plotReady = false;

    const el = {
      betaLabel: document.getElementById("betaLabel"),
      betaSlider: document.getElementById("betaSlider"),
      betaRail: document.getElementById("betaRail"),
      playBtn: document.getElementById("playBtn"),
      isoBtn: document.getElementById("isoBtn"),
      topBtn: document.getElementById("topBtn"),
      sideBtn: document.getElementById("sideBtn"),
      plot: document.getElementById("plot"),
    };

    function betaKey(beta) {
      return Number(beta).toFixed(2);
    }

    function fmt(value, digits = 2) {
      if (!Number.isFinite(Number(value))) return "--";
      return Number(value).toFixed(digits);
    }

    function nearestIndex(values, target) {
      let bestIndex = 0;
      let bestDistance = Infinity;
      values.forEach((value, i) => {
        const distance = Math.abs(Number(value) - Number(target));
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = i;
        }
      });
      return bestIndex;
    }

    function valueAtNearestGrid(surface, xValue, yValue) {
      const xIndex = nearestIndex(payload.x, xValue);
      const yIndex = nearestIndex(payload.y, yValue);
      return surface.z[yIndex][xIndex];
    }

    function axis3d(title, range) {
      return {
        title,
        autorange: false,
        range,
        backgroundcolor: "#f8faf7",
        gridcolor: "#d6dbd1",
        showbackground: true,
        zerolinecolor: "#aeb8a9",
      };
    }

    function renderSurface(nextIndex) {
      const surface = payload.surfaces[nextIndex];
      const fit = payload.fit[nextIndex] || surface;
      const wells = surface.wells || [];
      const wellX = wells.map((point) => point[0]);
      const wellY = wells.map((point) => point[1]);
      const wellZ = wells.map((point) => valueAtNearestGrid(surface, point[0], point[1]));
      const centerZ = valueAtNearestGrid(surface, 0, 0);

      const traces = [
        {
          type: "surface",
          x: payload.x,
          y: payload.y,
          z: surface.z,
          cmin: lossRange[0],
          cmax: lossRange[1],
          colorscale: "Portland",
          colorbar: { title: "proxy loss", len: 0.74, thickness: 14 },
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
          text: "beta " + betaKey(surface.beta) + " | Umax " + fmt(fit.Umax_total_energy || surface.Umax) + " | ridge " + fmt(fit.ridge_coef || surface.a) + " | center " + fmt(fit.center_coef || surface.c),
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
          camera: camera || {
            eye: { x: 1.55, y: 1.55, z: 0.92 },
            up: { x: 0, y: 0, z: 1 },
          },
          xaxis: axis3d("latent x", xyRange.x),
          yaxis: axis3d("latent y", xyRange.y),
          zaxis: axis3d("proxy loss", lossRange),
          aspectmode: "cube",
        },
        uirevision: "preserve-camera",
      };

      Plotly.react(el.plot, traces, layout, {
        displaylogo: false,
        responsive: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
      });

      if (!plotReady) {
        el.plot.on("plotly_relayout", (event) => {
          if (event["scene.camera"]) camera = event["scene.camera"];
        });
        plotReady = true;
      }
    }

    function setIndex(nextIndex) {
      index = Math.max(0, Math.min(nextIndex, payload.surfaces.length - 1));
      el.betaSlider.value = String(index);
      el.betaLabel.textContent = betaKey(payload.surfaces[index].beta);
      [...el.betaRail.querySelectorAll(".beta-dot")].forEach((button, i) => {
        button.classList.toggle("active", i === index);
      });
      renderSurface(index);
    }

    function setCamera(nextCamera) {
      camera = nextCamera;
      Plotly.relayout(el.plot, { "scene.camera": nextCamera });
    }

    function buildBetaRail() {
      el.betaRail.innerHTML = payload.surfaces
        .map((surface, i) => '<button class="beta-dot" data-index="' + i + '" title="beta ' + betaKey(surface.beta) + '">' + betaKey(surface.beta) + '</button>')
        .join("");
      [...el.betaRail.querySelectorAll(".beta-dot")].forEach((button) => {
        button.addEventListener("click", () => setIndex(Number(button.dataset.index)));
      });
    }

    el.betaSlider.max = String(payload.surfaces.length - 1);
    el.betaSlider.addEventListener("input", () => setIndex(Number(el.betaSlider.value)));
    el.playBtn.addEventListener("click", () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        el.playBtn.textContent = "Play";
        return;
      }
      el.playBtn.textContent = "Pause";
      timer = setInterval(() => {
        setIndex((index + 1) % payload.surfaces.length);
      }, 900);
    });
    el.isoBtn.addEventListener("click", () => setCamera({ eye: { x: 1.55, y: 1.55, z: 0.92 }, up: { x: 0, y: 0, z: 1 } }));
    el.topBtn.addEventListener("click", () => setCamera({ eye: { x: 0, y: 0, z: 2.25 }, up: { x: 0, y: 1, z: 0 } }));
    el.sideBtn.addEventListener("click", () => setCamera({ eye: { x: 2.35, y: 0.05, z: 0.35 }, up: { x: 0, y: 0, z: 1 } }));

    buildBetaRail();
    setIndex(0);
  </script>
</body>
</html>
`;
}

async function main() {
  fs.mkdirSync(VENDOR_DIR, { recursive: true });
  if (!fs.existsSync(PLOTLY_PATH)) {
    await download(PLOTLY_URL, PLOTLY_PATH);
  }

  const payloadText = fs.readFileSync(PAYLOAD_PATH, "utf8");
  const payload = JSON.parse(payloadText);
  const plotlyText = fs.readFileSync(PLOTLY_PATH, "utf8");
  const domains = computeSurfaceDomains(payload);
  const out = html({ payloadText, plotlyText, domains });
  fs.writeFileSync(OUT_HTML, out, "utf8");

  console.log(
    JSON.stringify(
      {
        output: OUT_HTML,
        bytes: Buffer.byteLength(out),
        embeddedPlotlyBytes: Buffer.byteLength(plotlyText),
        embeddedPayloadBytes: Buffer.byteLength(payloadText),
        lossRange: domains.lossRange,
        xyRange: domains.xyRange,
      },
      null,
      2
    )
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
