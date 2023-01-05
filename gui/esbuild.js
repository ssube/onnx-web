import { build } from 'esbuild';
import { join } from 'path';

const root = process.cwd();

build({
    bundle: true,
    define: {
        global: 'window',
    },
    entryPoints: [
        join(root, 'out/src/main.js'),
    ],
    keepNames: true,
    outdir: 'out/bundle/',
    platform: 'browser',
    sourcemap: true,
}).catch(() => process.exit(1));
