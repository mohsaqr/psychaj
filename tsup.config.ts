import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'core/index': 'src/core/index.ts',
    'estimation/index': 'src/estimation/index.ts',
    'models/index': 'src/models/index.ts',
    'validation/index': 'src/validation/index.ts',
    'graph/index': 'src/graph/index.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: true,
});
