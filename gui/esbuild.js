import { build, context } from 'esbuild';
import { join } from 'path';
import alias from 'esbuild-plugin-alias';
import { copy } from 'esbuild-plugin-copy';

function envTrue(key) {
    const val = (process.env[key] || '').toLowerCase();
    return val === '1' || val === 't' || val === 'true' || val === 'y' || val === 'yes';
}

const debug = envTrue('DEBUG');
const watch = envTrue('WATCH');
const root = process.cwd();

const plugins = [];

if (debug) {
    plugins.push(alias({
        'react-dom$': 'react-dom/profiling',
        'scheduler/tracing': 'scheduler/tracing-profiling',
    }));
}

const config = {
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
};

if (watch) {
    const copyArray = (files) => files.map(file =>
        copy({
            resolveFrom: 'cwd',
            assets: {
                from: [file],
                to: ['out/'],
            },
        })
    );

    const ctx = await context({
        ...config,
        entryPoints: [
            join(root, 'src/main.tsx'),
        ],
        plugins: [
            ...plugins,
            ...copyArray(['src/index.html', 'src/config.json']),
        ],
        banner: {
            js: `new EventSource('/esbuild').addEventListener('change', () => location.reload());`,
        },
    });

    await ctx.watch();

    const { host, port } = await ctx.serve({
        host: '127.0.0.1',
        port: 8000,
        servedir: 'out/',
    });

    console.log(`Serving on http://${host}:${port}`);
} else {
    build(config).catch(() => process.exit(1));
}