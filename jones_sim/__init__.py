"""Jones matrix forward modeling for radio interferometry."""

__version__ = "0.1.0"

# Core simulation components
# Calibration solver
from .antsol import AntSolSolver, solve_gains_from_ms
from .effects import (
    BandpassDelay,
    CrosshandPhase,
    ElectronicGains,
    InstrumentalLeakage,
    ParallacticAngle,
    RLDelayDifference,
    RotationMeasure,
)
from .simulator import JonesSimulator
from .source_models import (
    SourceModel,
    create_circular_source,
    create_linear_source,
    create_rm_source,
    create_unpolarized_source,
)
from .visibility_generator import VisibilityGenerator

# MS calibration (requires CASA)
try:
    from .ms_calibration import MSCalibrator, quick_gaincal  # noqa: F401

    MS_CALIBRATION_AVAILABLE = True
except ImportError:
    MS_CALIBRATION_AVAILABLE = False

# CASA interface (optional, requires casatools)
try:
    from .casa_interface import (  # noqa: F401
        CalibrationTableHandler,
        MeasurementSetHandler,
        quick_cal_summary,
        quick_ms_summary,
    )

    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False

__all__ = [
    # Version
    "__version__",
    # Simulation
    "JonesSimulator",
    "VisibilityGenerator",
    # Effects
    "ParallacticAngle",
    "ElectronicGains",
    "InstrumentalLeakage",
    "BandpassDelay",
    "RLDelayDifference",
    "CrosshandPhase",
    "RotationMeasure",
    # Source models
    "SourceModel",
    "create_unpolarized_source",
    "create_linear_source",
    "create_circular_source",
    "create_rm_source",
    # Calibration
    "AntSolSolver",
    "solve_gains_from_ms",
    # CASA (if available)
    "CASA_AVAILABLE",
    "MS_CALIBRATION_AVAILABLE",
]

if CASA_AVAILABLE:
    __all__.extend(
        [
            "MeasurementSetHandler",
            "CalibrationTableHandler",
            "quick_ms_summary",
            "quick_cal_summary",
        ]
    )

if MS_CALIBRATION_AVAILABLE:
    __all__.extend(["MSCalibrator", "quick_gaincal"])
