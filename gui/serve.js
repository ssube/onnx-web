import { mustDefault } from '@apextoaster/js-utils';
import { readFile } from 'fs';
import { createServer } from 'http';
import { join } from 'path';

const { env } = process;

const host = mustDefault(env.ONNX_WEB_DEV_HOST, '0.0.0.0');
const port = mustDefault(env.ONNX_WEB_DEV_PORT, '3000');
const root = process.cwd();

const portNum = parseInt(port, 10);

const server = createServer((req, res) => {
  readFile(join(root, 'out', req.url || 'index.html'), function (err, data) {
    if (err) {
      res.writeHead(404);
      res.end(JSON.stringify(err));
      return;
    }
    res.writeHead(200);
    res.end(data);
  });
});

server.listen(portNum, host, () => {
  console.log(`Dev server running at http://${host}:${port}/index.html`);
});
