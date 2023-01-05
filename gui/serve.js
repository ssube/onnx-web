import { readFile } from 'fs';
import { createServer } from 'http';
import { join } from 'path';

const hostname = '127.0.0.1';
const port = 3000;
const root = process.cwd();

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

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
