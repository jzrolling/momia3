"""Optional AI model backends (torch is an optional dependency).

Import a concrete backend explicitly, e.g.::

    from momia3.models.omnipose_lite import FastOmnipose

This package deliberately imports no torch-dependent submodule eagerly so the
core momia3 install stays lightweight.
"""

__all__: list[str] = []
