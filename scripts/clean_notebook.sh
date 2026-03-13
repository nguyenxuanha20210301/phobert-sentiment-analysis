#!/bin/bash
# Clean widget metadata from all notebooks so GitHub can render them
for nb in notebooks/*.ipynb; do
    python3 -c "
import nbformat, sys
path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
if 'widgets' in nb.metadata:
    del nb.metadata['widgets']
    nbformat.write(nb, path)
    print(f'✅ Cleaned: {path}')
else:
    print(f'⏭️  No widgets: {path}')
" "$nb"
done
