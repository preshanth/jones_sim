# Robust Cal Project - Claude Collaboration Guidelines

## Pair Programming Methodology

### Confidence-Driven Development
- **All statements must include confidence levels (High/Medium/Low)**
- **Confidence must be backed by specific data/evidence**
- **Format: Statement [Confidence: X% - Evidence: specific reference]**

### Evidence Sources Priority
1. Code inspection and analysis
2. Documentation references
3. Mathematical/theoretical foundations
4. Industry standards and practices
5. Reasonable inference from context

### Communication Standards
- Critical thinking over assumptions
- Question unclear requirements immediately
- Validate understanding before implementation
- Provide alternative approaches when confidence is low
- **Act as peer/pair programmer, not cheerleader**
- Avoid excessive praise ("Excellent!", "Perfect!", "Great!")
- Use neutral acknowledgments ("Done", "Updated", "Fixed")
- Be critical and point out issues, don't just validate
- Save enthusiasm for genuinely exceptional cases only

### Documentation Standards
- **NEVER oversell or inflate capabilities**
- State only what exists in code, not aspirations
- No marketing language ("powerful", "comprehensive", "advanced")
- Bad: "Symbolic equation generation - SymPy-based Jones matrix algebra"
- Good: "Symbolic Jones matrices - Generate Jones matrix equations using SymPy"
- Verify claims by checking actual code before writing
- When unsure, check the codebase rather than assume

### Decision Making
- High confidence (>80%): Proceed with implementation
- Medium confidence (50-80%): Discuss trade-offs and get user confirmation
- Low confidence (<50%): Research further or request clarification

## Project Context
- **Domain**: Radio astronomy calibration and Jones matrix modeling
- **Goal**: Complete Jones matrix forward modeling with symbolic equations, numerical simulation, and Monte Carlo error analysis
- **Current Status**: Core implementation complete, building visualization and Monte Carlo layers

## Project Progress Summary

### Completed Work
1. **Symbolic Layer (SymPy)**
   - JonesEquationGenerator class: symbolic matrix generation and simplification
   - All individual Jones matrices: P, E, X, G, B, T, R, C
   - Kronecker product implementation for Mueller matrices
   - Validation tests reproducing EVLA memo equations

2. **Numerical Simulation Layer**
   - Individual effect classes: ParallacticAngle, ElectronicGains, InstrumentalLeakage, etc.
   - JonesSimulator coordinator for effect chaining
   - Visibility corruption via Kronecker products
   - Complete test suite with >95% coverage

3. **Package Infrastructure**
   - Pip-installable package structure (jones_sim)
   - Proper dependencies and development tools
   - Documentation and parameter analysis (ParameterDistribution.md)
   - Mathematical validation against LaTeX specifications

4. **Effect Parameterization Analysis**
   - Time-varying effects: gains, parallactic angle, TEC
   - Frequency-varying effects: bandpass (jagged), R/L delays
   - Static effects: leakage, cross-hand phase, misalignment
   - Monte Carlo distribution framework designed

### Completed Phase: Visualization and Monte Carlo
1. **Interactive Plotting (Bokeh)**
   - Time-domain visualization: gains vs time with uncertainty bands
   - Frequency-domain visualization: jagged bandpass responses
   - Monte Carlo envelope plotting for parameter distributions
   - Effect isolation and combination analysis
   - Comprehensive dashboard with tabbed interface

2. **Parameter Distribution Classes**
   - Realistic statistical models for each effect type
   - Correlated parameter sampling (XX/YY gains, adjacent channels)
   - Time evolution models for thermal drift
   - Frequency correlation models for bandpass structure

3. **Monte Carlo Framework**
   - Unified PyMC sampler for complete Jones chains
   - Parameter space sampling and uncertainty propagation
   - Statistical analysis and effect summaries
   - Command-line interface with JSON configuration

### Completed Phase: Real Data Integration and Synthetic Visibility Pipeline
1. **Source Models and Visibility Generation (COMPLETED)**
   - **RotationMeasure effect class**: Missing Jones matrix implemented with λ² frequency dependence
   - **Source model classes**: 4 polarization scenarios (unpolarized, linear, RM-affected, circular+linear)
   - **Visibility corruption pipeline**: Complete Stokes → correlations → Jones corruption → noise
   - **Flag handling**: Proper flag propagation through entire processing chain
   - **Realistic noise model**: Complex Gaussian with σ_real = σ_imag = σ_total/√2

2. **CASA Interface (COMPLETED)**
   - **MeasurementSetHandler**: Read/write MS data with proper flag handling using casatools
   - **CalibrationTableHandler**: Read gain/bandpass solutions from CASA calibration tables
   - **Flag-aware processing**: All operations preserve flag structure (True = flagged/bad data)
   - **Realistic test parameters**: VLA L-band, ALMA Band 6, compact array configurations
   - **No mock tests**: Only real functionality testing with actual casatools

3. **Comprehensive Validation Framework (COMPLETED)**
   - **4 source corruption cases**: Unpolarized, 5% linear@30°, RM=25 rad/m², 10% circular+2% linear
   - **Systematic effect isolation**: Individual and combined Jones effect analysis
   - **Realistic observational parameters**: Integration times, solution intervals, flagging patterns
   - **Complete test coverage**: 95%+ with pytest validation of all components

### Current Status: Ready for Real Data Validation
**Implementation Complete**: Full synthetic visibility generation pipeline with CASA integration
**Next Phase**: Validation with actual measurement sets and calibration tables

### Technical Implementation Details
**Corruption Pipeline**: Ideal Stokes → Linear Correlations → Apply Jones → Add Noise → Corrupted Visibilities
**Recovery Pipeline**: Corrupted Visibilities → Fit Jones Model → Parameter Estimates → Compare to Truth
**Validation Focus**: Calibrator sources, direction-independent, point source assumption
**Coordinate System**: XY linear feeds with parallactic angle rotation
**Scope**: Start with gains + bandpass, expand incrementally to other effects

### Key Validation Questions
- Can we predict gain thermal drift from visibility time series?
- Can we detect bandpass structure from frequency-dependent corruption?
- What are detection thresholds for different effect magnitudes?
- How do different array configurations affect error sensitivity?

### Key Implementation Notes
- All Jones matrices are 2×2 complex (even amplitude/phase converted to complex)
- Order matters: P·E·X·G·B·T·R·C (parallactic angle first, cross-hand phase last)
- Package name: `jones_sim`
- Kronecker product produces 4×4 Mueller operating on [XX,XY,YX,YY]
- XY linear feed coordinates (not HV circular)
- Focus on unknown telescope design guidance and realistic error regimes
- Least code path and solutions. Solutions over fixes. Always run everything by me. Particularly when you can't figure it out on your own.