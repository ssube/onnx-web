import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { join } from 'path';
import alias from 'esbuild-plugin-alias';

const require = createRequire(import.meta.url);
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
    plugins: [
        alias({
            'react-dom$': 'react-dom/profiling',
            'scheduler/tracing': 'scheduler/tracing-profiling',
        })
    ],
    sourcemap: true,
}).catch(() => process.exit(1));
