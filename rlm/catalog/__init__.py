"""Dataset catalog. Drop one ``*.py`` module per dataset into this package
and expose a module-level ``DESCRIPTOR: DatasetDescriptor``. The registry
picks them up automatically at server startup.
"""

from __future__ import annotations
