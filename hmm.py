#!/usr/bin/env python
"""Entry point kept for backwards compatibility: ``python hmm.py`` still works.

The implementation now lives in the ``regime_allocation`` package; run
``python hmm.py --help`` to see the available options.
"""

import sys

from regime_allocation.cli import main

if __name__ == "__main__":
    sys.exit(main())
