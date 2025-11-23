# Jones Simulator Documentation

Comprehensive API and user documentation for the Jones Simulator.

## Building Documentation

### Install Dependencies

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated documentation will be in `docs/_build/html/`.

### View Documentation

```bash
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

### Clean Build

```bash
make clean
```

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation entry point
- `architecture.rst` - System architecture overview
- `workflows.rst` - Common usage patterns
- `configuration.rst` - JSON configuration guide
- `contributing_docs.rst` - Guide for maintaining documentation
- `api/` - API reference documentation
  - `core.rst` - Core modules (simulator, config, interfaces)
  - `effects.rst` - Jones effects and samplers
  - `calibration.rst` - Calibration solvers
  - `utilities.rst` - Plotting and helper functions
- `ARCHITECTURE.md` - Detailed system design document
- `Makefile` - Build commands
- `requirements.txt` - Documentation dependencies

## Contributing

See `contributing_docs.rst` for detailed guidelines on:

- Writing docstrings (Google style)
- Adding new modules to API reference
- Building and testing documentation locally
- Mathematical notation with LaTeX
- Cross-referencing and linking

## Quick Tips

**Add a new module to docs:**
1. Write comprehensive docstrings in the code
2. Add module to appropriate `api/*.rst` file
3. Rebuild: `make html`
4. Check for warnings

**Test documentation changes:**
```bash
make clean && make html
```

**Common issues:**
- Module not found: Check `autodoc_mock_imports` in `conf.py`
- Warnings: Fix docstring formatting, add missing references
- Build errors: Check RST syntax, ensure files referenced exist

## Live Rebuild (Optional)

For development, use auto-rebuild:

```bash
pip install sphinx-autobuild
make livehtml
```

Opens browser with auto-reloading on file changes.
