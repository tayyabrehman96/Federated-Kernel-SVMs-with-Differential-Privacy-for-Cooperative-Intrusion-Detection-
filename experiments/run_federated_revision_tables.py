#!/usr/bin/env python3
"""
Regenerate federated synthetic metrics and LaTeX helper rows for the MDPI revision.

Delegates to generate_revision_results.main(). From repo root:
  python experiments/run_federated_revision_tables.py

Or (recommended):
  cd experiments && python run_federated_revision_tables.py
"""
from generate_revision_results import main

if __name__ == "__main__":
    main()
