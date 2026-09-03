"""Time-Series Operator (TSO).

A from-scratch prototype of a foundation-model style pipeline for time series:

    signal -> Takens embedding (phase space) -> Koopman lift (linearization)
             -> forecast + attractor geometry

No code is shared with any other project in this workspace.
"""

from . import attractors, embedding, koopman, neural_field, pipeline, viz

__all__ = ["attractors", "embedding", "koopman", "neural_field", "pipeline", "viz"]
__version__ = "0.1.0"
