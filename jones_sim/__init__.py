"""Jones matrix forward modeling for radio interferometry."""

__version__ = "0.1.0"

# Core simulation components
from .simulator import JonesSimulator
from .effects import (
    ParallacticAngle,
    ElectronicGains,
    InstrumentalLeakage,
    BandpassDelay,
    RLDelayDifference,
    CrosshandPhase,
    RotationMeasure,
)
from .source_models import (
    SourceModel,
    create_unpolarized_source,
    create_linear_source,
    create_circular_source,
    create_rm_source,
)
from .visibility_generator import VisibilityGenerator

# Calibration solver
from .antsol import AntSolSolver, solve_gains_from_ms

# MS calibration (requires CASA)
try:
    from .ms_calibration import MSCalibrator, quick_gaincal

    MS_CALIBRATION_AVAILABLE = True
except ImportError:
    MS_CALIBRATION_AVAILABLE = False

# CASA interface (optional, requires casatools)
try:
    from .casa_interface import (
        MeasurementSetHandler,
        CalibrationTableHandler,
        quick_ms_summary,
        quick_cal_summary,
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
