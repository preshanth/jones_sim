#!/usr/bin/env python3
"""ValidationPipeline class for end-to-end Jones calibration validation.

Design principles:
- Each effect is independent and resumable
- State is saved after each step
- Comparison returns dicts (no printing)
- Plotting is optional (default=True)
"""

import os
import numpy as np
from typing import Dict, List, Optional

from jones_sim import BandpassDelay, ElectronicGains, InstrumentalLeakage
from jones_sim.calibration_solver import CalibrationSolver
from jones_sim.casa_interface import MeasurementSetHandler
from jones_sim.plotting_enhanced import (
    plot_three_way_comparison,
    plot_bandpass_comparison,
    plot_leakage_dterms,
)

# Import validation utilities
import sys
sys.path.insert(0, os.path.dirname(__file__))
from validation_lib import (
    generate_delays,
    generate_gains,
    generate_bandpass,
    generate_dterms,
    compute_parallactic_angles,
    corrupt_ms,
    run_casa_calibration,
    read_casa_delays,
    read_casa_gains,
    read_casa_bandpass,
    read_casa_dterms,
    save_ground_truth,
    load_ground_truth,
)


class ValidationPipeline:
    """Manages end-to-end Jones calibration validation pipeline.

    Each effect (K, B, G, D) goes through:
    1. Ground truth generation
    2. MS corruption
    3. CASA calibration
    4. Our calibration
    5. Comparison
    6. Plotting (optional)

    State is saved after each step for resume capability.
    """

    def __init__(
        self,
        ms_path: str,
        output_dir: str = ".",
        effects: List[str] = None,
        seed: int = 100,
    ):
        """Initialize validation pipeline.

        Args:
            ms_path: Path to measurement set
            output_dir: Directory for output files
            effects: List of effects to validate (default: K, B, G, D)
            seed: Random seed for ground truth generation
        """
        self.ms_path = ms_path
        self.output_dir = output_dir
        self.effects = effects or ["K", "B", "G", "D"]
        self.seed = seed

        # State
        self.ground_truth = {}
        self.casa_solutions = {}
        self.our_solutions = {}
        self.comparison_results = {}

        # File paths
        self.base_name = os.path.basename(ms_path).replace(".ms", "")
        self.truth_file = os.path.join(output_dir, f"{self.base_name}_truth.npz")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def generate_ground_truth(
        self,
        n_channels: int = 64,
        delay_range_ns: float = 10.0,
        amp_std: float = 0.1,
        phase_std: float = 0.1,
        bandpass_delay_range_ns: float = 5.0,
        bandpass_amp_variation: float = 0.05,
        leakage_level: float = 0.05,
    ) -> Dict:
        """Generate ground truth for all effects.

        Returns:
            Dictionary with ground truth values
        """
        print(f"\nGenerating ground truth for effects: {', '.join(self.effects)}")

        # Get MS info
        ms_handler = MeasurementSetHandler(self.ms_path)
        summary = ms_handler.get_observation_summary()
        n_antennas = summary["n_antennas"]
        freqs = summary["frequency_info"][0]["chan_freqs"]
        ms_handler.close()

        truth = {
            "n_antennas": n_antennas,
            "n_channels": n_channels,
            "freqs": freqs,
        }

        # K: Delays
        if "K" in self.effects:
            delays_ns, delays_sec = generate_delays(
                n_antennas, delay_range_ns, seed=self.seed
            )
            truth["K_delays_ns"] = delays_ns
            truth["K_delays_sec"] = delays_sec
            print(f"  K: ±{delay_range_ns} ns")

        # B: Bandpass
        if "B" in self.effects:
            bandpass, bp_delays = generate_bandpass(
                n_antennas, n_channels, freqs,
                delay_range_ns=bandpass_delay_range_ns,
                amp_variation=bandpass_amp_variation,
                seed=self.seed + 1,
            )
            truth["B_bandpass"] = bandpass
            truth["B_delays_sec"] = bp_delays
            print(f"  B: ±{bandpass_delay_range_ns} ns delay, {bandpass_amp_variation} amp var")

        # G: Gains
        if "G" in self.effects:
            gains = generate_gains(
                n_antennas, amp_std=amp_std, phase_std=phase_std, seed=self.seed + 2
            )
            truth["G_gains"] = gains
            print(f"  G: amp_std={amp_std}, phase_std={phase_std}")

        # D: Leakage
        if "D" in self.effects:
            dterms = generate_dterms(
                n_antennas, leakage_level=leakage_level, seed=self.seed + 3
            )
            truth["D_dterms"] = dterms
            print(f"  D: leakage={leakage_level}")

        self.ground_truth = truth
        save_ground_truth(self.truth_file, truth)

        return truth

    def corrupt_ms(self, add_noise: bool = True, sefd: float = 420.0) -> None:
        """Corrupt MS with all effects.

        Args:
            add_noise: Add thermal noise
            sefd: System equivalent flux density (Jy)
        """
        print(f"\nCorrupting MS with effects: {', '.join(self.effects)}")

        truth = self.ground_truth
        freqs = truth["freqs"]
        effects = {}

        # K: Delays
        if "K" in self.effects:
            delay_effect = BandpassDelay(
                tau_xx=truth["K_delays_sec"],
                tau_yy=truth["K_delays_sec"],
                ref_freq=0.0,
            )
            effects["delays"] = delay_effect  # Use 'delays' to match effect_order

        # B: Bandpass (frequency-dependent gains)
        if "B" in self.effects:
            bandpass = truth["B_bandpass"]

            def bp_xx(freq, time, ant_id):
                freq_idx = np.argmin(np.abs(freqs - freq))
                return bandpass[ant_id, 0, freq_idx]

            def bp_yy(freq, time, ant_id):
                freq_idx = np.argmin(np.abs(freqs - freq))
                return bandpass[ant_id, 1, freq_idx]

            bandpass_effect = ElectronicGains(g_xx=bp_xx, g_yy=bp_yy)
            effects["bandpass"] = bandpass_effect  # Use 'bandpass' to match effect_order

        # G: Gains
        if "G" in self.effects:
            gains = truth["G_gains"]
            gain_effect = ElectronicGains(g_xx=gains[:, 0], g_yy=gains[:, 1])
            effects["gains"] = gain_effect  # Use 'gains' to match effect_order

        # D: Leakage (with parallactic angle rotation P → D)
        if "D" in self.effects:
            from jones_sim import ParallacticAngle
            from casatools import table

            # Get MS metadata for parang computation
            ms_handler = MeasurementSetHandler(self.ms_path)
            summary = ms_handler.get_observation_summary()
            n_antennas = summary["n_antennas"]

            # Get observation times
            tb = table()
            tb.open(self.ms_path)
            times_mjd = np.unique(tb.getcol("TIME"))
            tb.close()

            # Get source declination from FIELD table
            tb.open(self.ms_path + "/FIELD")
            phase_dir = tb.getcol("PHASE_DIR")[:, :, 0]  # Field 0
            source_dec_deg = np.rad2deg(phase_dir[1, 0])
            tb.close()

            # VLA latitude (approx)
            latitude_deg = 34.0784

            # Convert times to hours from transit (simplified - assume first time is transit)
            times_hours = (times_mjd - times_mjd[0]) / 3600.0

            # Compute parallactic angles
            parang_array = compute_parallactic_angles(
                times_hours, latitude_deg, source_dec_deg, n_antennas
            )  # [n_ant, n_time]

            truth["D_parang"] = parang_array

            # Create parang effect - callable for time-dependent angles
            def get_parang(time, ant_id):
                idx = np.argmin(np.abs(times_mjd - time))
                return parang_array[ant_id, idx]

            # P rotates sky Q,U into feed frame
            parang_effect = ParallacticAngle(angles=get_parang)
            effects["parallactic"] = parang_effect

            # D-terms are instrumental leakage in feed frame
            dterms = truth["D_dterms"]

            def d_hv(freq, time, ant_id):
                return dterms[ant_id, 0]

            def d_vh(freq, time, ant_id):
                return dterms[ant_id, 1]

            leakage_effect = InstrumentalLeakage(d_hv=d_hv, d_vh=d_vh, theta=0.0)
            effects["leakage"] = leakage_effect

            # Note: Data stays in feed frame, no P^-1 needed
            # CASA polcal uses multiple parang values to separate D from source Q,U

        corrupt_ms(self.ms_path, effects, add_noise=add_noise, sefd=sefd)

    def run_casa_calibration(self, refant: str = "0") -> Dict:
        """Run sequential CASA calibration for all effects.

        Order: K → B → G → D (standard CASA order)

        Args:
            refant: Reference antenna

        Returns:
            Dictionary mapping effect names to caltable paths
        """
        print(f"\nRunning CASA calibration: {' → '.join(self.effects)}")

        caltables = {}
        gaintables = []  # Cumulative for sequential correction

        for effect in self.effects:
            caltable = os.path.join(
                self.output_dir,
                f"{self.base_name}_casa.{effect}"
            )

            print(f"\n  Solving {effect}...")
            run_casa_calibration(
                ms_path=self.ms_path,
                cal_type=effect,
                caltable=caltable,
                refant=refant,
                solint="inf",
                gaintables=gaintables,
            )

            caltables[effect] = caltable
            gaintables.append(caltable)

        self.casa_solutions = caltables
        return caltables

    def run_our_calibration(self, effect_name: str, spw: str = "0"):
        """Run our CalibrationSolver for one effect.

        Args:
            effect_name: Effect to solve (K, B, G, or D)
            spw: Spectral window selection

        Returns:
            CalibrationSolver instance, or None if not supported
        """
        print(f"\n  Solving {effect_name} with our solver...")

        solver = CalibrationSolver(self.ms_path)
        solver.load_data(spw=spw, solint="inf")

        # Apply prior corrections
        prior_effects = self.effects[:self.effects.index(effect_name)]
        for prior in prior_effects:
            if prior in self.our_solutions and self.our_solutions[prior] is not None:
                prior_solution = self.our_solutions[prior].get_solution(prior)
                solver.apply_corrections(**{prior: prior_solution})

        # Add effect and load CASA solution
        if effect_name == "K":
            solver.add_effect("K", solint="inf", prior_bound_ns=0.2)
        elif effect_name == "B":
            solver.add_effect("B", solint="inf", calmode="ap")
        elif effect_name == "G":
            solver.add_effect("G", solint="inf", calmode="ap", prior_std=0.3)
        elif effect_name == "D":
            solver.add_effect("D", solint="inf", calmode="ap", prior_std=0.01)

        solver.load_casa_solutions(**{effect_name: self.casa_solutions[effect_name]})
        solver.build_model()

        # Optimize
        print(f"  Running MAP optimization for {effect_name}...")
        solver.optimize(num_steps=1000)
        solver.print_summary()

        # Save
        solver_file = os.path.join(
            self.output_dir,
            f"{self.base_name}_{effect_name}_solver.npz"
        )
        solution = solver.get_solution(effect_name)
        np.savez(solver_file, solution=solution, **solver.trace)
        print(f"  ✓ Saved: {solver_file}")

        self.our_solutions[effect_name] = solver
        return solver

    def compare_effect(self, effect_name: str) -> Dict:
        """Compare truth vs CASA vs ours for one effect.

        Args:
            effect_name: Effect to compare

        Returns:
            Dictionary with comparison results
        """
        truth = self.ground_truth
        n_antennas = truth["n_antennas"]

        if effect_name == "K":
            # Read CASA delays
            casa_delays_ns = read_casa_delays(
                self.casa_solutions["K"], n_antennas
            )

            # Get our delays
            solver = self.our_solutions["K"]
            delays_free = solver.trace["delays_free"]
            our_delays_ns = np.zeros(n_antennas)
            our_delays_ns[0] = 0.0  # Reference
            for ant in range(1, n_antennas):
                our_delays_ns[ant] = np.mean(delays_free[:, ant - 1]) * 1e9

            # Compare using validation_lib function
            truth_delays_ns = truth["K_delays_ns"]
            from validation_lib import compare_delays
            return compare_delays(truth_delays_ns, casa_delays_ns, our_delays_ns)

        elif effect_name == "G":
            # Read CASA gains
            casa_gains = read_casa_gains(self.casa_solutions["G"], n_antennas)

            # Get our gains
            solver = self.our_solutions["G"]
            our_gains = solver.get_solution("G")

            # Compare
            truth_gains = truth["G_gains"]
            from validation_lib import compare_gains
            return compare_gains(truth_gains, casa_gains, our_gains)

        elif effect_name == "B":
            # Read CASA bandpass
            n_channels = truth["n_channels"]
            casa_bp = read_casa_bandpass(self.casa_solutions["B"], n_antennas, n_channels)

            # Get our bandpass
            solver = self.our_solutions["B"]
            our_bp_raw = solver.get_solution("B")

            # Solver returns [n_ant, n_chan, 2], we need [n_ant, 2, n_chan]
            our_bp = our_bp_raw.transpose(0, 2, 1)

            # Compare
            truth_bp = truth["B_bandpass"]
            freqs = truth["freqs"]
            from validation_lib import compare_bandpass
            return compare_bandpass(truth_bp, casa_bp, our_bp, freqs)

        elif effect_name == "D":
            # Read CASA D-terms
            casa_dterms = read_casa_dterms(self.casa_solutions["D"], n_antennas)

            # Get our D-terms
            solver = self.our_solutions["D"]
            our_dterms = solver.get_solution("D")

            # Compare
            truth_dterms = truth["D_dterms"]
            from validation_lib import compare_leakage
            return compare_leakage(truth_dterms, casa_dterms, our_dterms)

        return {}

    def plot_effect(self, effect_name: str) -> str:
        """Generate comparison plot for one effect.

        Args:
            effect_name: Effect to plot

        Returns:
            Path to generated plot file
        """
        results = self.comparison_results.get(effect_name)
        if not results:
            print(f"  No comparison results for {effect_name}")
            return None

        plot_file = os.path.join(
            self.output_dir,
            f"{self.base_name}_{effect_name}_comparison.html"
        )

        if effect_name == "K":
            plot_three_way_comparison(
                results["truth"],
                results["casa"],
                results["ours"],
                title="Delay (K) Validation",
                x_label="Antenna",
                y_label="Delay (ns)",
                output_file_path=plot_file,
            )

        elif effect_name == "G":
            # Plot gains amplitude comparison
            truth_amp = np.abs(results["truth"][:, 0])  # XX pol
            casa_amp = np.abs(results["casa"][:, 0])
            ours_amp = np.abs(results["ours"][:, 0])

            plot_three_way_comparison(
                truth_amp,
                casa_amp,
                ours_amp,
                title="Gain Amplitude (G) Validation - XX Polarization",
                x_label="Antenna",
                y_label="Amplitude",
                output_file_path=plot_file,
            )

        elif effect_name == "B":
            # Use specialized bandpass plot - plot first antenna, XX pol
            plot_bandpass_comparison(
                results["freqs"],
                truth_bp=results["truth"],
                casa_bp=results["casa"],
                recovered_bp=results["ours"],
                antenna_idx=1,  # Plot antenna 1 (not ref)
                pol_idx=0,      # XX polarization
                output_file_path=plot_file,
            )

        elif effect_name == "D":
            # Plot D-term magnitude comparison
            truth_mag = np.abs(results["truth"][:, 0])  # d_xy
            casa_mag = np.abs(results["casa"][:, 0])
            ours_mag = np.abs(results["ours"][:, 0])

            plot_three_way_comparison(
                truth_mag,
                casa_mag,
                ours_mag,
                title="D-term Magnitude (D) Validation - d_xy",
                x_label="Antenna",
                y_label="|d_xy|",
                output_file_path=plot_file,
            )

        print(f"  ✓ Plot saved: {plot_file}")
        return plot_file

    def run(
        self,
        resume: bool = False,
        generate_plots: bool = True,
        add_noise: bool = True,
    ) -> Dict:
        """Run full validation pipeline.

        Args:
            resume: Resume from saved state
            generate_plots: Generate comparison plots
            add_noise: Add thermal noise to MS

        Returns:
            Dictionary with all results
        """
        # Load or generate ground truth
        if resume and os.path.exists(self.truth_file):
            print(f"Resuming from {self.truth_file}")
            self.ground_truth = load_ground_truth(self.truth_file)
        else:
            self.generate_ground_truth()
            self.corrupt_ms(add_noise=add_noise)

        # CASA calibration
        self.run_casa_calibration()

        # Our calibration (one effect at a time)
        for effect in self.effects:
            self.run_our_calibration(effect)

            # Compare
            results = self.compare_effect(effect)
            self.comparison_results[effect] = results

            # Plot
            if generate_plots:
                self.plot_effect(effect)

        return self.comparison_results

    def save_state(self) -> None:
        """Save complete pipeline state."""
        state_file = os.path.join(self.output_dir, f"{self.base_name}_state.npz")
        np.savez(
            state_file,
            effects=self.effects,
            **self.ground_truth,
            **self.comparison_results,
        )
        print(f"✓ State saved: {state_file}")

    def load_state(self) -> None:
        """Load pipeline state."""
        state_file = os.path.join(self.output_dir, f"{self.base_name}_state.npz")
        if os.path.exists(state_file):
            data = np.load(state_file, allow_pickle=True)
            self.effects = list(data["effects"])
            self.ground_truth = {k: data[k] for k in data.files if k != "effects"}
            print(f"✓ State loaded: {state_file}")
