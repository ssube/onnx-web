import { mustDefault } from '@apextoaster/js-utils';
import { readFile } from 'fs';
import { createServer } from 'http';
import { join } from 'path';

const { env } = process;

const host = mustDefault(env.ONNX_WEB_DEV_HOST, '127.0.0.1');
const port = mustDefault(env.ONNX_WEB_DEV_PORT, '8000');
const root = process.cwd();

const portNum = parseInt(port, 10);

const contentTypes = [
  [/^.*\.html$/, 'text/html'],
  [/^.*\.js$/, 'application/javascript'],
  [/^.*\.json$/, 'text/json'],
];

function getContentType(path) {
  for (const [regex, type] of contentTypes) {
    if (regex.test(path)) {
      return type;
    }
  }

  return 'unknown';
}

const server = createServer((req, res) => {
  const path = join(root, 'out', req.url || 'index.html');
  readFile(path, function (err, data) {
    if (err) {
      res.writeHead(404);
      res.end(JSON.stringify(err));
      return;
    }
    res.writeHead(200, {
      'Content-Type': getContentType(path),
    });
    res.end(data);
  });
});

server.listen(portNum, host, () => {
  console.log(`Dev server running at http://${host}:${port}/index.html`);
});
