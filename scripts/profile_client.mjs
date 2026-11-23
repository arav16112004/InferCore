// One persistent HTTP connection per caller. No third-party Node dependencies.
import http from 'node:http';
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
const { port, concurrency, direct, cases, warmup, duration } =
  JSON.parse(readFileSync(0, 'utf8'));
const agents = Array.from({ length: concurrency }, () => new http.Agent({
  keepAlive: true, maxSockets: 1, maxFreeSockets: 1,
}));

function send(agent, body) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      host: '127.0.0.1', port, path: '/infer', agent, method: 'POST',
      headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) },
    }, res => {
      let content = '';
      res.setEncoding('utf8');
      res.on('data', chunk => { content += chunk; });
      res.on('end', () => resolve({ res, content }));
      res.on('error', reject);
    });
    req.setTimeout(3000, () => req.destroy(new Error('transport_timeout')));
    req.on('error', reject);
    req.end(body);
  });
}

async function phase(seconds) {
  const rows = [];
  const start = performance.now();
  const until = start + 1000 * seconds;
  await Promise.all(agents.map(async (agent, lane) => {
    let seq = 0;
    while (performance.now() < until) {
      const item = cases[(lane + seq++) % cases.length];
      const input = direct
        ? { values: item.values, remaining_ms: 1000, request_id: lane + '-' + seq }
        : { values: item.values, deadline_ms: 1000 };
      const sent = performance.now();
      let status = 0, queueUs = null, workerQueueUs = null, ortUs = null, worker = null, error = null;
      try {
        const { res, content } = await send(agent, JSON.stringify(input));
        status = res.statusCode;
        if (status === 200) {
          const { output, backend, worker_id, request_id } = JSON.parse(content);
          if (!Array.isArray(output) || output.length !== 2 ||
              !output.every((p, i) => Number.isFinite(p) && Math.abs(p - item.expected[i]) <= 2e-6) ||
              (!direct && (backend !== 'onnx' || !Number.isInteger(worker_id) ||
                        !request_id || res.headers['x-request-id'] !== request_id))) {
            throw new Error('invalid_prediction_or_identity');
          }
          worker = direct ? 0 : worker_id;
          const header = res.headers['x-queue-wait-us'];
          if (header !== undefined) queueUs = Number(header);
          if (res.headers['x-worker-queue-us'] !== undefined) workerQueueUs = Number(res.headers['x-worker-queue-us']);
          if (res.headers['x-onnx-run-us'] !== undefined) ortUs = Number(res.headers['x-onnx-run-us']);
        }
      } catch (e) {
        status = 0;
        error = e.code || e.message;
      }
      rows.push([sent-start, performance.now()-sent, status, queueUs, worker, error, workerQueueUs, ortUs]);
    }
  }));
  return { rows, elapsed_s: (performance.now()-start)/1000 };
}

try {
  await phase(warmup);
  process.stdout.write(JSON.stringify(await phase(duration)));
} finally {
  agents.forEach(agent => agent.destroy());
}
