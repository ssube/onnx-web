import { build, context } from 'esbuild';
import { join } from 'path';
import alias from 'esbuild-plugin-alias';
import { copy } from 'esbuild-plugin-copy';

const debug = process.env['DEBUG'] === 'TRUE';
const watch = process.env['WATCH'] === 'TRUE';
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
            ...copyArray(['src/index.html', 'examples/config.json']),
        ],
        banner: {
            js: `new EventSource('/esbuild').addEventListener('change', () => location.reload());`,
        },
    });

    await ctx.watch();

    const { host, port } = await ctx.serve({
        servedir: 'out/',
    });

    console.log(`Serving on http://${host}:${port}`);
} else {
    build(config).catch(() => process.exit(1));
}