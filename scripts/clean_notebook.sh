#!/bin/bash
# Usage: ./scripts/clean_notebook.sh notebooks/01_data_pipeline.ipynb
python3 -c "
import nbformat, sys
path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
if 'widgets' in nb.metadata:
    del nb.metadata['widgets']
nbformat.write(nb, path)
print(f'✅ Cleaned: {path}')
" "$1"
