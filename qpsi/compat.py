"""Numpy compatibility shims for numpy 1.26 (qpsi-temp) vs 2.x (default env)."""

import numpy as np

# np.trapezoid was added in numpy 2.0; numpy 1.26 has np.trapz
trapezoid = getattr(np, "trapezoid", None) or np.trapz
