import { build } from 'esbuild';
import { join } from 'path';
import alias from 'esbuild-plugin-alias';

const debug = process.env['DEBUG'] === 'TRUE'
const root = process.cwd();

const plugins = [];

if (debug) {
    plugins.push(alias({
        'react-dom$': 'react-dom/profiling',
        'scheduler/tracing': 'scheduler/tracing-profiling',
    }));
}

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
    plugins,
    sourcemap: true,
}).catch(() => process.exit(1));
