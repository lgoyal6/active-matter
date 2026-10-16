/* Renders the reviewed 32-run APC summary from the checked-in study data.
   This page is a recorded-study explorer and does not execute the model. */
const LAW_LABELS = {
  thesis_eq5_5_normalized: 'Thesis Eq. 5.5, normalized',
  zenodo_ch6_source_unnormalized: 'Zenodo Chapter 6, source law'
};
const palette = ['#527b62', '#b76843', '#5979b8', '#9868a3'];
const dataPromise = fetch('./data.json').then(response => {
  if (!response.ok) throw new Error(`Study data unavailable (${response.status})`);
  return response.json();
});

function svgElement(svg, name, attrs = {}, label = '') {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (label) node.textContent = label;
  svg.append(node);
  return node;
}

function renderChart(data, law) {
  const svg = document.querySelector('#density-chart');
  const rows = data.rows.filter(row => row.law === law).sort((a, b) => a.rho - b.rho);
  const min = Math.min(0, ...rows.flatMap(row => row.seed_slopes)) - 0.03;
  const max = Math.max(...rows.flatMap(row => row.seed_slopes)) + 0.03;
  const left = 75;
  const right = 770;
  const top = 34;
  const bottom = 325;
  const x = index => left + index * (right - left) / Math.max(1, rows.length - 1);
  const y = value => bottom - (value - min) * (bottom - top) / (max - min);
  svg.replaceChildren();
  svgElement(svg, 'title', {}, `${LAW_LABELS[law]} contact rate by density`);
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = min + (max - min) * tick / 4;
    const py = y(value);
    svgElement(svg, 'line', { x1: left, y1: py, x2: right, y2: py, class: 'grid-line' });
    svgElement(svg, 'text', { x: left - 12, y: py + 4, 'text-anchor': 'end', class: 'axis-label' }, value.toFixed(2));
  }
  svgElement(svg, 'line', { x1: left, y1: bottom, x2: right, y2: bottom, class: 'axis-line' });
  rows.forEach((row, index) => {
    const cx = x(index);
    const color = palette[index % palette.length];
    const spreadTop = y(row.mean + row.sample_sd);
    const spreadBottom = y(row.mean - row.sample_sd);
    svgElement(svg, 'line', { x1: cx, y1: spreadTop, x2: cx, y2: spreadBottom, class: 'sd-line', stroke: color });
    svgElement(svg, 'line', { x1: cx - 8, y1: spreadTop, x2: cx + 8, y2: spreadTop, class: 'sd-cap', stroke: color });
    svgElement(svg, 'line', { x1: cx - 8, y1: spreadBottom, x2: cx + 8, y2: spreadBottom, class: 'sd-cap', stroke: color });
    row.seed_slopes.forEach((value, seedIndex) => {
      const offset = (seedIndex - 1.5) * 8;
      svgElement(svg, 'circle', { cx: cx + offset, cy: y(value), r: 4, fill: color, class: 'seed-dot' });
    });
    svgElement(svg, 'circle', { cx, cy: y(row.mean), r: 6, fill: '#17231e', stroke: '#fff', 'stroke-width': 2 });
    svgElement(svg, 'text', { x: cx, y: bottom + 25, 'text-anchor': 'middle', class: 'axis-label' }, `ρ ${row.rho}`);
    svgElement(svg, 'text', { x: cx, y: spreadTop - 10, 'text-anchor': 'middle', class: 'mean-label' }, row.mean.toFixed(3));
  });
  svg.setAttribute('aria-label', `${LAW_LABELS[law]} contact-rate slopes across four densities. Each density shows four seed values and a mean with sample standard deviation.`);
}

function renderTable(data, law) {
  const body = document.querySelector('#density-table');
  body.replaceChildren();
  for (const row of data.rows.filter(item => item.law === law).sort((a, b) => a.rho - b.rho)) {
    const tr = document.createElement('tr');
    const values = row.seed_slopes.map(value => value.toFixed(3)).join(', ');
    tr.innerHTML = `<td>${row.rho}</td><td>${row.mean.toFixed(4)}</td><td>${row.sample_sd.toFixed(4)}</td><td>${row.se.toFixed(4)}</td><td class="seed-values">${values}</td>`;
    body.append(tr);
  }
}

dataPromise.then(data => {
  document.querySelector('#cases').textContent = data.study.raw_curves_verified;
  document.querySelector('#groups').textContent = data.rows.length;
  document.querySelector('#seeds').textContent = data.study.seeds.length;
  document.querySelector('#steps').textContent = `${(data.study.steps_per_case / 1000).toFixed(0)}K`;
  document.querySelector('#metric-note').textContent = `${data.study.metric} The vertical bars show one sample standard deviation across the four seed runs. Points show the individual seeds.`;
  const select = document.querySelector('#law');
  for (const law of data.study.laws) {
    const option = document.createElement('option');
    option.value = law;
    option.textContent = LAW_LABELS[law];
    select.append(option);
  }
  select.addEventListener('change', () => {
    renderChart(data, select.value);
    renderTable(data, select.value);
  });
  const limits = document.querySelector('#limits');
  for (const value of data.limits) {
    const item = document.createElement('li');
    item.textContent = value;
    limits.append(item);
  }
  document.querySelector('#provenance').textContent = `Study source SHA-256 ${data.source.study_source_sha256}; protocol ${data.source.protocol_sha256}; independent math review ${data.source.math_review_sha256}.`;
  document.querySelector('#source-ref').textContent = `main ${data.source.commit.slice(0, 7)} · 32 cases`;
  renderChart(data, select.value);
  renderTable(data, select.value);
}).catch(error => {
  document.querySelector('.hero .eyebrow').textContent = `Study data could not be loaded: ${error.message}`;
});
