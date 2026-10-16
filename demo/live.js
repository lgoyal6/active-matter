/* Calls the bounded Python simulator and renders its returned trajectory.
 * The frame slider shows computed positions; it does not run a second model.
 * A failed or cancelled request clears the prior result instead of replaying it.
 */
const $ = id => document.getElementById(id);
let result = null;
let controller = null;
function clearResult() {
  result = null;
  $('frame').disabled = true;
  $('metrics').replaceChildren();
  $('receipt').textContent = 'No current result.';
  $('frame-label').textContent = '';
  $('swarm').getContext('2d').clearRect(0, 0, 600, 600);
}
function draw() {
  if (!result) return;
  const index = Number($('frame').value);
  const ctx = $('swarm').getContext('2d');
  ctx.clearRect(0, 0, 600, 600);
  const scale = 600 / result.box_size;
  ctx.fillStyle = '#c2672f';
  ctx.beginPath(); ctx.arc(result.source_position[0] * scale, result.source_position[1] * scale, 8, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#236044';
  for (const [x, y] of result.frames[index]) {
    ctx.beginPath(); ctx.arc(x * scale, y * scale, 3, 0, Math.PI * 2); ctx.fill();
  }
  $('frame-label').textContent = `Step ${result.frame_steps[index]} of ${result.inputs.steps - 1}; orange marks the source.`;
}
$('frame').addEventListener('input', draw);
$('cancel').addEventListener('click', () => controller?.abort());
$('experiment').addEventListener('submit', async event => {
  event.preventDefault();
  if (controller) return;
  clearResult();
  controller = new AbortController();
  const requestController = controller;
  const timer = setTimeout(() => requestController.abort(), 20000);
  $('run').disabled = true; $('cancel').disabled = false;
  $('status').textContent = 'Computing a new trajectory with the Python engine...';
  try {
    const inputs = Object.fromEntries([...new FormData(event.target)].map(([key, value]) => [key, Number(value)]));
    const response = await fetch('/api/simulate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(inputs), signal: requestController.signal });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
    result = body;
    $('frame').max = result.frames.length - 1; $('frame').value = 0; $('frame').disabled = false;
    const metrics = [['Final mean distance to source (model units)', result.metrics.final_mean_distance.toFixed(3)], ['Final heading alignment (0-1)', result.metrics.final_alignment.toFixed(3)], ['Computed steps', result.inputs.steps], ['Computed particles', result.inputs.particles]];
    for (const [label, value] of metrics) {
      const dt = document.createElement('dt'); dt.textContent = label;
      const dd = document.createElement('dd'); dd.textContent = String(value);
      $('metrics').append(dt, dd);
    }
    $('receipt').textContent = JSON.stringify({mode: result.mode, inputs: result.inputs, engine_source_sha256: result.engine_source_sha256, trajectory_sha256: result.trajectory_sha256}, null, 2);
    $('status').textContent = 'New trajectory computed. Move the frame slider to inspect it.';
    draw();
  } catch (error) {
    clearResult();
    $('status').textContent = error.name === 'AbortError' ? 'Request cancelled or timed out. No saved result was substituted.' : `Simulation failed: ${error.message}`;
  } finally {
    clearTimeout(timer); controller = null;
    $('run').disabled = false; $('cancel').disabled = true;
  }
});
