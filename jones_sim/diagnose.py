#!/usr/bin/env python3
"""Diagnostic script to examine MS data and understand noise characteristics."""


import numpy as np

try:
    from casa_interface import MeasurementSetHandler

    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False
    print("ERROR: casa_interface not available")
    exit(1)


def diagnose_ms_noise(ms_path, cal_table=None, spw=0, field=0, max_vis=50000):
    """Examine MS data to understand noise characteristics."""

    print(f"\n{'=' * 80}")
    print("MS NOISE DIAGNOSTICS")
    print(f"{'=' * 80}")
    print(f"MS: {ms_path}")
    print(f"Cal table: {cal_table}")

    # Open MS
    ms_handler = MeasurementSetHandler(ms_path)
    summary = ms_handler.get_observation_summary()
    n_antennas = summary["n_antennas"]

    print(f"\nAntennas: {n_antennas}")
    print(f"SPWs: {summary['n_spw']}")
    print(f"Fields: {summary['field_names']}")

    # Read DATA
    print(f"\nReading SPW {spw}, Field {field}...")
    data_dict = ms_handler.read_visibilities(field=field, spw=spw)

    data_array = data_dict["data"]  # (n_corr, n_chan, n_row)
    antenna1 = data_dict["antenna1"]
    antenna2 = data_dict["antenna2"]

    if "frequency" in data_dict:
        freqs = data_dict["frequency"]
    else:
        freqs = summary["frequency_info"][spw]["chan_freqs"]

    n_corr, n_chan, n_row = data_array.shape
    print(f"DATA shape: {n_corr}×{n_chan}×{n_row}")

    # Read MODEL_DATA
    import casatools

    tb = casatools.table()
    tb.open(ms_path)
    model_array = tb.getcol("MODEL_DATA")
    tb.close()

    print(f"MODEL_DATA shape: {model_array.shape}")

    # Handle shape mismatch
    if data_array.shape != model_array.shape:
        min_row = min(data_array.shape[2], model_array.shape[2])
        data_array = data_array[:, :, :min_row]
        model_array = model_array[:, :, :min_row]
        antenna1 = antenna1[:min_row]
        antenna2 = antenna2[:min_row]
        n_row = min_row

    # Flatten
    obs_list, model_list, freq_list, a1_list, a2_list = [], [], [], [], []

    for row in range(n_row):
        for chan in range(n_chan):
            obs_list.append(data_array[:, chan, row])
            model_list.append(model_array[:, chan, row])
            freq_list.append(freqs[chan])
            a1_list.append(antenna1[row])
            a2_list.append(antenna2[row])

            if len(obs_list) >= max_vis:
                break
        if len(obs_list) >= max_vis:
            break

    observed_vis = np.array(obs_list, dtype=complex)  # (n_vis, 4)
    model_vis = np.array(model_list, dtype=complex)
    frequencies = np.array(freq_list)
    antenna1 = np.array(a1_list, dtype=int)
    antenna2 = np.array(a2_list, dtype=int)
    # n_vis = len(observed_vis)

    print(f"\n{'=' * 80}")
    print("AMPLITUDE STATISTICS")
    print(f"{'=' * 80}")

    # Compute amplitudes
    obs_amp = np.abs(observed_vis)
    model_amp = np.abs(model_vis)

    print("\nObserved amplitudes:")
    print(f"  Mean: {np.mean(obs_amp):.6f} Jy")
    print(f"  Std: {np.std(obs_amp):.6f} Jy")
    print(f"  Min: {np.min(obs_amp):.6f} Jy")
    print(f"  Max: {np.max(obs_amp):.6f} Jy")
    print(f"  Median: {np.median(obs_amp):.6f} Jy")

    print("\nModel amplitudes:")
    print(f"  Mean: {np.mean(model_amp):.6f} Jy")
    print(f"  Std: {np.std(model_amp):.6f} Jy")
    print(f"  Min: {np.min(model_amp):.6f} Jy")
    print(f"  Max: {np.max(model_amp):.6f} Jy")

    print("\nAmplitude ratio (obs/model):")
    amp_ratio = obs_amp / (model_amp + 1e-10)
    print(f"  Mean: {np.mean(amp_ratio):.6f}")
    print(f"  Std: {np.std(amp_ratio):.6f}")
    print("  This should be ~1.0 if delays don't affect amplitude")

    # Compute phases
    obs_phase = np.angle(observed_vis)  # (n_vis, 4) in radians
    model_phase = np.angle(model_vis)

    print(f"\n{'=' * 80}")
    print("PHASE STATISTICS (UNCORRECTED)")
    print(f"{'=' * 80}")

    print("\nObserved phases:")
    print(f"  Mean: {np.mean(obs_phase):.3f} rad")
    print(f"  Std: {np.std(obs_phase):.3f} rad")

    print("\nModel phases:")
    print(f"  Mean: {np.mean(model_phase):.3f} rad")
    print(f"  Std: {np.std(model_phase):.3f} rad")

    # Phase difference (raw)
    phase_diff_raw = obs_phase - model_phase

    # Wrap to [-π, π]
    phase_diff_wrapped = np.angle(np.exp(1j * phase_diff_raw))

    print("\nPhase residual (obs - model):")
    print(f"  Mean: {np.mean(phase_diff_wrapped):.3f} rad")
    print(f"  Std: {np.std(phase_diff_wrapped):.3f} rad")
    print(f"  Min: {np.min(phase_diff_wrapped):.3f} rad")
    print(f"  Max: {np.max(phase_diff_wrapped):.3f} rad")
    print("  Large std indicates delay corruption!")

    # Read CASA delays if available
    if cal_table:
        print(f"\n{'=' * 80}")
        print("APPLYING CASA DELAYS")
        print(f"{'=' * 80}")

        tb.open(cal_table)
        fparam = tb.getcol("FPARAM")
        flags = tb.getcol("FLAG")
        antennas = tb.getcol("ANTENNA1")
        tb.close()

        casa_delays_ns = np.zeros(n_antennas)
        delay_counts = np.zeros(n_antennas)

        if fparam.ndim == 3:
            n_pol, n_chan_cal, n_rows = fparam.shape
            for row in range(n_rows):
                ant = antennas[row]
                if ant < n_antennas:
                    for pol in range(n_pol):
                        for chan in range(n_chan_cal):
                            if not flags[pol, chan, row]:
                                casa_delays_ns[ant] += fparam[pol, chan, row]
                                delay_counts[ant] += 1
        elif fparam.ndim == 2:
            n_pol, n_rows = fparam.shape
            for row in range(n_rows):
                ant = antennas[row]
                if ant < n_antennas:
                    for pol in range(n_pol):
                        if not flags[pol, row]:
                            casa_delays_ns[ant] += fparam[pol, row]
                            delay_counts[ant] += 1

        mask = delay_counts > 0
        casa_delays_ns[mask] /= delay_counts[mask]
        casa_delays = casa_delays_ns * 1e-9  # Convert to seconds

        print("\nCASA delays (ns):")
        for ant in range(min(n_antennas, 10)):
            print(f"  Ant {ant}: {casa_delays_ns[ant]:.3f}")

        # Apply delays - TEST BOTH SIGNS
        tau1 = casa_delays[antenna1]
        tau2 = casa_delays[antenna2]

        # Sign 1: tau1 - tau2
        phase_correction_1 = 2 * np.pi * (tau1 - tau2) * frequencies

        # Sign 2: tau2 - tau1
        phase_correction_2 = 2 * np.pi * (tau2 - tau1) * frequencies

        print(f"\n{'=' * 80}")
        print("PHASE RESIDUALS AFTER CASA CORRECTION")
        print(f"{'=' * 80}")

        # Test sign 1
        phase_diff_corrected_1 = phase_diff_wrapped - phase_correction_1[:, None]
        phase_diff_corrected_1 = np.angle(np.exp(1j * phase_diff_corrected_1))

        print("\nSign convention 1: φ = 2π(tau1 - tau2)ν")
        print(f"  Mean residual: {np.mean(phase_diff_corrected_1):.6f} rad")
        print(f"  Std residual: {np.std(phase_diff_corrected_1):.6f} rad")
        print(f"  RMS residual: {np.sqrt(np.mean(phase_diff_corrected_1**2)):.6f} rad")

        # Test sign 2
        phase_diff_corrected_2 = phase_diff_wrapped - phase_correction_2[:, None]
        phase_diff_corrected_2 = np.angle(np.exp(1j * phase_diff_corrected_2))

        print("\nSign convention 2: φ = 2π(tau2 - tau1)ν")
        print(f"  Mean residual: {np.mean(phase_diff_corrected_2):.6f} rad")
        print(f"  Std residual: {np.std(phase_diff_corrected_2):.6f} rad")
        print(f"  RMS residual: {np.sqrt(np.mean(phase_diff_corrected_2**2)):.6f} rad")

        # Pick the better one
        rms1 = np.sqrt(np.mean(phase_diff_corrected_1**2))
        rms2 = np.sqrt(np.mean(phase_diff_corrected_2**2))

        if rms1 < rms2:
            print("\n✓ Sign 1 is correct (tau1 - tau2)")
            phase_diff_corrected = phase_diff_corrected_1
            phase_correction = phase_correction_1
        else:
            print("\n✓ Sign 2 is correct (tau2 - tau1)")
            phase_diff_corrected = phase_diff_corrected_2
            phase_correction = phase_correction_2

        # Phase noise estimate
        phase_noise_std = np.std(phase_diff_corrected)

        print(f"\n{'=' * 80}")
        print("NOISE ESTIMATES")
        print(f"{'=' * 80}")

        print("\nPhase noise:")
        print(
            f"  Std: {phase_noise_std:.6f} rad = {np.degrees(phase_noise_std):.3f} deg"
        )

        # Convert to amplitude noise
        # For Gaussian noise: σ_phase ≈ σ_amp / |V|
        mean_amp = np.mean(obs_amp)
        amplitude_noise_from_phase = phase_noise_std * mean_amp

        print("\nImplied amplitude noise:")
        print("  σ_amp ≈ σ_phase × |V|")
        print(f"  σ_amp ≈ {phase_noise_std:.6f} × {mean_amp:.3f}")
        print(f"  σ_amp ≈ {amplitude_noise_from_phase:.6f} Jy")

        # Also compute amplitude scatter directly
        cos_corr = np.cos(phase_correction[:, None])
        sin_corr = np.sin(phase_correction[:, None])

        predicted_real = model_vis.real * cos_corr - model_vis.imag * sin_corr
        predicted_imag = model_vis.real * sin_corr + model_vis.imag * cos_corr

        residual_real = observed_vis.real - predicted_real
        residual_imag = observed_vis.imag - predicted_imag

        amplitude_noise_direct = np.std(
            np.concatenate([residual_real.flatten(), residual_imag.flatten()])
        )

        print("\nDirect amplitude noise estimate:")
        print(
            f"  Std of (obs - model) after CASA correction: {amplitude_noise_direct:.6f} Jy"
        )

        print("\nComparison:")
        print(f"  From phase: {amplitude_noise_from_phase:.6f} Jy")
        print(f"  From amplitude: {amplitude_noise_direct:.6f} Jy")
        print(f"  Ratio: {amplitude_noise_direct / amplitude_noise_from_phase:.3f}")
        print("  (Should be ~1.0 if noise is Gaussian)")

        # SNR analysis
        print(f"\n{'=' * 80}")
        print("SNR ANALYSIS")
        print(f"{'=' * 80}")

        snr = obs_amp / amplitude_noise_direct

        print("\nSNR statistics:")
        print(f"  Mean: {np.mean(snr):.1f}")
        print(f"  Median: {np.median(snr):.1f}")
        print(f"  Min: {np.min(snr):.1f}")
        print(f"  Max: {np.max(snr):.1f}")

        low_snr_frac = np.sum(snr < 3) / snr.size
        print(f"  Fraction with SNR < 3: {low_snr_frac * 100:.1f}%")
        print("  (Low SNR → non-Gaussian phase noise!)")

        # Correlation analysis
        print(f"\n{'=' * 80}")
        print("CORRELATION ANALYSIS")
        print(f"{'=' * 80}")

        # Average over correlations
        phase_diff_avg = np.mean(phase_diff_corrected, axis=1)  # (n_vis,)

        # Group by baseline
        unique_baselines = []
        for a1, a2 in zip(antenna1, antenna2):
            bl = (min(a1, a2), max(a1, a2))
            if bl not in unique_baselines:
                unique_baselines.append(bl)

        print(f"\nNumber of unique baselines: {len(unique_baselines)}")

        # Phase residual vs frequency (check for residual delays)
        from scipy import stats

        # Fit line to phase vs frequency for each baseline
        print("\nChecking for residual delay trends (phase vs freq):")

        n_test_baselines = min(5, len(unique_baselines))
        for i, (a1, a2) in enumerate(unique_baselines[:n_test_baselines]):
            mask = (antenna1 == a1) & (antenna2 == a2)
            if np.sum(mask) > 10:
                f = frequencies[mask]
                p = phase_diff_avg[mask]

                # Linear fit: phase = delay * 2πν + offset
                slope, intercept, r_value, p_value, std_err = stats.linregress(f, p)

                residual_delay_ns = slope / (2 * np.pi) * 1e9

                print(
                    f"  Baseline {a1}-{a2}: residual delay = {residual_delay_ns:.3f} ns (R²={r_value**2:.3f})"
                )

        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")

        print("\n✓ Use phase-only likelihood with:")
        print(f"  σ_phase = {phase_noise_std:.6f} rad")
        print("  OR")
        print(f"  σ_amplitude = {amplitude_noise_direct:.6f} Jy")

        print("\n✓ Filter visibilities with SNR < 3 for stable phase estimates")

        print("\n✓ Phase wrapping: use np.angle() to wrap residuals to [-π, π]")

    ms_handler.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose MS noise characteristics")
    parser.add_argument("ms", help="Path to measurement set")
    parser.add_argument("--cal-table", help="CASA K-table with delays")
    parser.add_argument("--spw", type=int, default=0, help="Spectral window")
    parser.add_argument("--field", type=int, default=0, help="Field ID")
    parser.add_argument("--max-vis", type=int, default=50000, help="Max visibilities")

    args = parser.parse_args()

    diagnose_ms_noise(args.ms, args.cal_table, args.spw, args.field, args.max_vis)
