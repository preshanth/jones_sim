Contributing to Documentation
==============================

This guide explains how to build and maintain the Jones Simulator documentation.

Building Documentation
----------------------

Install dependencies::

    pip install -r docs/requirements.txt

Build HTML documentation::

    cd docs
    make html

View documentation::

    open _build/html/index.html

Clean build artifacts::

    make clean

Docstring Style Guide
---------------------

We use Google-style docstrings with type hints.

Functions
~~~~~~~~~

::

    def calibrate(
        data: np.ndarray,
        method: str = "map",
        threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """Calibrate visibility data using specified method.

        Args:
            data: Input visibility array of shape (n_vis, n_freq, n_pol).
            method: Calibration method, either "map" or "mcmc".
            threshold: Convergence threshold for optimization.

        Returns:
            Dictionary containing:
                - solutions: Calibrated Jones matrices
                - residuals: Fit residuals
                - converged: Boolean convergence flag

        Raises:
            ValueError: If method is not recognized.

        Examples:
            >>> result = calibrate(vis, method="map")
            >>> jones = result["solutions"]
        """

Classes
~~~~~~~

::

    class JonesEffect:
        """Base class for Jones matrix effects.

        Represents a single Jones matrix effect in the calibration chain.
        All effect types inherit from this base class.

        Attributes:
            params: Effect parameters dictionary.
            name: Human-readable effect name.

        Examples:
            >>> effect = BandpassEffect(params)
            >>> jones = effect.compute_jones(times, freqs, ant1, ant2)
        """

        def __init__(self, params: Dict[str, Any]):
            """Initialize Jones effect.

            Args:
                params: Effect parameters including distribution specs.
            """

Mathematical Notation
~~~~~~~~~~~~~~~~~~~~~

Use LaTeX for equations::

    """Compute Jones matrix for bandpass effect.

    The bandpass Jones matrix is:

    .. math::

        J_i(\nu) = \begin{pmatrix}
            g_{ix}(\nu) & 0 \\
            0 & g_{iy}(\nu)
        \end{pmatrix}

    where $g_{ip}(\nu)$ is the complex gain for antenna $i$,
    polarization $p$, and frequency $\nu$.
    """

Adding New Modules
------------------

When adding a new Python module:

1. Write comprehensive docstrings
2. Add module to appropriate API reference file in ``docs/api/``
3. Rebuild documentation to verify
4. Check for warnings or missing docstrings

Autodoc will automatically extract:

- Class and function signatures
- Docstrings
- Type hints
- Class inheritance

Best Practices
--------------

Documentation Quality
~~~~~~~~~~~~~~~~~~~~~

- Write clear, concise docstrings for all public APIs
- Include usage examples in docstrings
- Document all parameters, returns, and exceptions
- Use type hints consistently
- Explain the "why", not just the "what"

Code Examples
~~~~~~~~~~~~~

- Provide working code examples
- Show common use cases
- Include expected output when helpful
- Keep examples minimal and focused

Cross-References
~~~~~~~~~~~~~~~~

Link to related functions/classes::

    """See also :func:`solve_k` for delay calibration."""
    """Uses :class:`JonesEffect` as base class."""

External Links
~~~~~~~~~~~~~~

::

    """Based on the measurement equation from
    `Hamaker et al. (1996) <https://doi.org/10.1051/aas:1996146>`_.
    """

Documenting New Features
-------------------------

When implementing a new feature:

1. Write docstrings as you code
2. Add usage examples to ``workflows.rst``
3. Update configuration guide if adding new config options
4. Add to API reference if new module
5. Update main ``index.rst`` if major feature

Common Issues
-------------

Module Not Found
~~~~~~~~~~~~~~~~

If autodoc can't find a module, check:

- Module is in ``jones_sim/`` package
- ``__init__.py`` imports are correct
- Path in ``docs/conf.py`` is correct

Import Errors
~~~~~~~~~~~~~

Mock imports for optional dependencies in ``conf.py``::

    autodoc_mock_imports = ["casatools", "casatasks"]

Warnings
~~~~~~~~

Fix all Sphinx warnings before committing. Common issues:

- Missing references
- Malformed docstrings
- Duplicate labels

Continuous Integration
----------------------

Documentation is built automatically in CI. Check for:

- Build succeeds without errors
- No warnings in build log
- All cross-references resolve
- Examples run without errors

Local Testing
-------------

Before pushing changes::

    # Clean build
    cd docs
    make clean
    make html

    # Check for warnings
    # Fix any issues reported

    # View locally
    open _build/html/index.html

Directory Structure
-------------------

::

    docs/
    ├── conf.py              # Sphinx configuration
    ├── index.rst            # Main documentation index
    ├── architecture.rst     # System architecture (converted from .md)
    ├── workflows.rst        # Common usage patterns
    ├── configuration.rst    # Configuration guide
    ├── contributing_docs.rst # This file
    ├── api/                 # API reference
    │   ├── core.rst
    │   ├── effects.rst
    │   ├── calibration.rst
    │   └── utilities.rst
    ├── Makefile             # Build commands
    ├── requirements.txt     # Doc build dependencies
    └── _build/              # Generated HTML (gitignored)

Updating Architecture Docs
---------------------------

The ``ARCHITECTURE.md`` file contains detailed system design.

To update::

    # Edit the markdown file
    vim docs/ARCHITECTURE.md

    # Reference in RST files
    # Convert if needed for Sphinx inclusion

Resources
---------

- Sphinx documentation: https://www.sphinx-doc.org/
- reStructuredText primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
- Google docstring style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- NumPy docstring style: https://numpydoc.readthedocs.io/en/latest/format.html
