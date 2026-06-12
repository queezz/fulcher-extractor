def test_import_package():
    import fulcher_extractor

    assert fulcher_extractor is not None


def test_h2_line_database_matches_archaeology_shape():
    from fulcher_extractor.line_database import load_lines, wavelength_matrix

    lines = load_lines()
    h2 = [line for line in lines if line.isotopologue == "H2"]
    matrix = wavelength_matrix(lines, isotopologue="H2")

    assert len(h2) == 42
    assert matrix.shape == (11, 4)
    assert matrix.loc[1, "0-0"] == 601.8299
    assert matrix.loc[11, "1-1"] == 628.9709
    assert matrix.loc[7, "2-2"] == 629.6622
    assert matrix.loc[9, "3-3"] == 644.1498


def test_synthetic_single_gaussian_area_recovery():
    import numpy as np

    from fulcher_extractor.fit import FitConfig, fit_single_line
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.spectrocube_io import Spectrum

    rng = np.random.default_rng(42)
    wavelength = np.linspace(612.0, 612.4, 401)
    expected_area = 0.125
    baseline = 3.2
    intensity = baseline + gaussian_area_model(wavelength, expected_area, 612.1787, 0.045)
    intensity += rng.normal(0.0, 0.01, size=wavelength.size)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    line = FulcherLine(
        isotopologue="H2",
        branch="Q",
        v_upper=1,
        v_lower=1,
        band="1-1",
        N=1,
        wavelength_nm=612.1787,
        source_table="synthetic",
        original_unit="nm",
    )

    result = fit_single_line(
        spectrum,
        line,
        config=FitConfig(line_half_width_nm=0.18, initial_sigma_nm=0.05),
    )

    assert result.success
    assert result.baseline_strategy == "local_minimum"
    assert result.baseline_offset > baseline - 0.1
    assert abs(result.amplitude - expected_area) / expected_area < 0.08
    assert abs(result.center_nm - line.wavelength_nm) < 0.005


def test_global_dx_recenters_fit_expectation():
    import numpy as np

    from fulcher_extractor.fit import FitConfig, fit_single_line
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.spectrocube_io import Spectrum

    wavelength = np.linspace(612.0, 612.4, 401)
    rest_nm = 612.1787
    dx_nm = -0.035
    intensity = 1.0 + gaussian_area_model(wavelength, 0.11, rest_nm + dx_nm, 0.045)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    line = FulcherLine(
        isotopologue="H2",
        branch="Q",
        v_upper=1,
        v_lower=1,
        band="1-1",
        N=1,
        wavelength_nm=rest_nm,
        source_table="synthetic",
        original_unit="nm",
    )

    result = fit_single_line(
        spectrum,
        line,
        config=FitConfig(global_dx_nm=dx_nm, center_offset_nm=0.03),
    )

    assert result.success
    assert result.global_dx_nm == dx_nm
    assert abs(result.expected_center_nm - (rest_nm + dx_nm)) < 1e-12
    assert abs(result.center_offset_from_rest_nm - dx_nm) < 0.002
    assert abs(result.center_offset_from_expected_nm) < 0.002


def test_h2_scan_score_combines_fulcher_peaks_and_halpha():
    import numpy as np

    from fulcher_extractor.h2_dataset_cli import _score_scan_frame
    from fulcher_extractor.line_models import gaussian_area_model

    wavelength = np.linspace(598.0, 658.0, 3001)
    baseline = 0.02 + 0.001 * np.sin(wavelength)
    localized_fulcher = gaussian_area_model(wavelength, 0.010, 612.1787, 0.035)
    halpha = gaussian_area_model(wavelength, 0.030, 656.28, 0.040)
    spectrum = baseline + localized_fulcher + halpha

    score = _score_scan_frame(
        wavelength,
        spectrum,
        fulcher_min_nm=600.0,
        fulcher_max_nm=630.0,
    )

    assert score["scan_score"] > score["fulcher_signal"]
    assert score["fulcher_peak_count"] >= 1
    assert score["fulcher_peak_signal"] > score["fulcher_signal"]
    assert score["halpha_signal"] > 0
    assert score["scan_score_reason"] == "fulcher_peaks+halpha"


def test_instrument_width_restricts_sigma_bounds():
    import numpy as np

    from fulcher_extractor.fit import FitConfig, fit_single_line
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.spectrocube_io import Spectrum

    wavelength = np.linspace(622.3, 622.7, 401)
    rest_nm = 622.4815
    instrument_sigma = 0.0273
    intensity = 1.0 + gaussian_area_model(wavelength, 0.10, rest_nm, instrument_sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    line = FulcherLine(
        isotopologue="H2",
        branch="Q",
        v_upper=2,
        v_lower=2,
        band="2-2",
        N=1,
        wavelength_nm=rest_nm,
        source_table="synthetic",
        original_unit="nm",
    )

    result = fit_single_line(
        spectrum,
        line,
        config=FitConfig(
            instrument_sigma_nm=instrument_sigma,
            instrument_sigma_leeway_nm=0.015,
        ),
    )

    assert result.success
    assert result.sigma_lower_bound_nm == 0.015
    assert abs(result.sigma_upper_bound_nm - 0.0423) < 1e-12
    assert abs(result.sigma_nm - instrument_sigma) < 0.001


def test_estimate_instrument_width_filters_bad_lines():
    from fulcher_extractor.fit import LineFitResult, estimate_instrument_width

    def result(line_id, sigma, status="ok", relerr=0.03):
        return LineFitResult(
            line_id=line_id,
            isotopologue="H2",
            branch="Q",
            band="0-0",
            N=1,
            rest_wavelength_nm=601.0,
            amplitude=1.0,
            amplitude_stderr=relerr,
            center_nm=601.0,
            center_stderr_nm=0.0,
            sigma_nm=sigma,
            sigma_stderr_nm=0.0,
            fwhm_nm=2.354820045 * sigma,
            baseline_offset=0.0,
            baseline_strategy="local_minimum",
            window_min_nm=600.8,
            window_max_nm=601.2,
            background_min_nm=599.5,
            background_max_nm=602.5,
            n_points=20,
            residual_rms=0.0,
            success=True,
            status=status,
        )

    estimate = estimate_instrument_width(
        [
            result("good1", 0.026),
            result("good2", 0.028),
            result("wide", 0.16, "ok;sigma_at_upper_bound"),
            result("noisy", 0.03, relerr=0.5),
        ]
    )

    assert estimate.n_lines == 2
    assert estimate.sigma_nm == 0.027


def test_close_database_neighbours_fit_as_blend_group():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.spectrocube_io import Spectrum

    wavelength = np.linspace(623.55, 624.05, 501)
    line_a = FulcherLine("H2", "Q", 1, 1, "1-1", 9, 623.7457, "synthetic", "nm")
    line_b = FulcherLine("H2", "Q", 2, 2, "2-2", 3, 623.8391, "synthetic", "nm")
    sigma = 0.0273
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.08, line_a.wavelength_nm, sigma)
    intensity += gaussian_area_model(wavelength, 0.03, line_b.wavelength_nm, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    results = extract_lines(
        spectrum,
        lines=[line_a, line_b],
        isotopologue="H2",
        wavelength_min_nm=623.0,
        wavelength_max_nm=624.0,
        config=FitConfig(
            instrument_sigma_nm=sigma,
            instrument_sigma_leeway_nm=0.005,
            close_neighbor_threshold_nm=0.10,
        ),
    )

    assert len(results) == 2
    assert {result.line_id for result in results} == {line_a.line_id, line_b.line_id}
    assert all(result.blend_component_count == 2 for result in results)
    assert all("blend_group" in result.status for result in results)
    by_id = {result.line_id: result for result in results}
    assert by_id[line_a.line_id].amplitude > by_id[line_b.line_id].amplitude
    fitted_spacing = by_id[line_b.line_id].center_nm - by_id[line_a.line_id].center_nm
    database_spacing = line_b.wavelength_nm - line_a.wavelength_nm
    assert abs(fitted_spacing - database_spacing) < 1e-12
    assert by_id[line_a.line_id].blend_delta_nm == by_id[line_b.line_id].blend_delta_nm


def test_legacy_policy_zeroes_rejected_line_for_matrix_export():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.output import results_to_dataframe, results_to_matrices
    from fulcher_extractor.spectrocube_io import Spectrum

    line = FulcherLine("H2", "Q", 1, 1, "1-1", 5, 615.9565, "synthetic", "nm")
    wavelength = np.linspace(615.75, 616.15, 401)
    intensity = 1.0 + gaussian_area_model(wavelength, 0.13, line.wavelength_nm, 0.0273)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    result = extract_lines(
        spectrum,
        lines=[line],
        wavelength_min_nm=615.0,
        wavelength_max_nm=616.5,
        config=FitConfig(instrument_sigma_nm=0.0273, instrument_sigma_leeway_nm=0.005),
    )[0]
    report = results_to_dataframe([result])
    intensity_matrix, error_matrix = results_to_matrices([result])

    assert result.success
    assert result.amplitude > 0.12
    assert result.legacy_policy == "reject"
    assert result.legacy_matrix_action == "zero"
    assert "too big" in result.legacy_evidence
    assert "legacy_zeroed" in result.status
    assert report.loc[0, "amplitude"] > 0.12
    assert report.loc[0, "matrix_amplitude"] == 0.0
    assert intensity_matrix.loc[5, "1-1"] == 0.0
    assert error_matrix.loc[5, "1-1"] == 0.0


def test_legacy_policy_marks_accepted_blend_components():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.spectrocube_io import Spectrum

    line_a = FulcherLine("H2", "Q", 1, 1, "1-1", 9, 623.7457, "synthetic", "nm")
    line_b = FulcherLine("H2", "Q", 2, 2, "2-2", 3, 623.8391, "synthetic", "nm")
    wavelength = np.linspace(623.55, 624.05, 501)
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.08, line_a.wavelength_nm, 0.0273)
    intensity += gaussian_area_model(wavelength, 0.03, line_b.wavelength_nm, 0.0273)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    results = extract_lines(
        spectrum,
        lines=[line_a, line_b],
        wavelength_min_nm=623.0,
        wavelength_max_nm=624.0,
        config=FitConfig(
            instrument_sigma_nm=0.0273,
            instrument_sigma_leeway_nm=0.005,
            close_neighbor_threshold_nm=0.10,
        ),
    )

    assert {result.legacy_policy for result in results} == {"blend_accept"}
    assert {result.legacy_matrix_action for result in results} == {"keep"}
    assert all(result.legacy_line_scale_role == "used" for result in results)
    assert all(result.matrix_amplitude == result.amplitude for result in results)
    assert all("legacy_blend_accept" in result.status for result in results)


def test_boltzmann_policy_excludes_wide_line_without_zeroing_matrix():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.output import results_to_dataframe, results_to_matrices
    from fulcher_extractor.spectrocube_io import Spectrum

    line = FulcherLine("H2", "Q", 1, 1, "1-1", 2, 612.7246, "synthetic", "nm")
    wavelength = np.linspace(612.50, 612.95, 451)
    intensity = 1.0 + gaussian_area_model(wavelength, 0.08, line.wavelength_nm, 0.037)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    result = extract_lines(
        spectrum,
        lines=[line],
        wavelength_min_nm=612.0,
        wavelength_max_nm=613.5,
        config=FitConfig(instrument_sigma_nm=0.0273, instrument_sigma_leeway_nm=0.015),
    )[0]
    report = results_to_dataframe([result])
    intensity_matrix, error_matrix = results_to_matrices([result])

    assert result.success
    assert result.legacy_policy == "wide_peak_suspected_contamination"
    assert result.legacy_matrix_action == "keep"
    assert result.boltzmann_fit_action == "exclude"
    assert "wide" in result.boltzmann_fit_reason
    assert "boltzmann_excluded" in result.status
    assert report.loc[0, "boltzmann_fit_action"] == "exclude"
    assert intensity_matrix.loc[2, "1-1"] == result.matrix_amplitude
    assert intensity_matrix.loc[2, "1-1"] > 0.0
    assert error_matrix.loc[2, "1-1"] > 0.0


def test_overview_qc_lines_include_trusted_decontaminated_lines():
    from fulcher_extractor.line_database import load_lines
    from fulcher_extractor.line_policy import overview_qc_lines

    labels = {line.line_id for line in overview_qc_lines(load_lines())}

    assert {"H2_Q3_0-0", "H2_Q6_1-1", "H2_Q2_2-2"}.issubset(labels)
    assert "H2_Q7_0-0" in labels
    assert "H2_Q4_1-1" not in labels


def test_clean_sigma_mask_excludes_rejected_suspicious_failed_and_bound_rows():
    import pandas as pd

    from fulcher_extractor.sigma_stats import clean_sigma_mask, summarize_sigma

    rows = [
        {
            "line_id": "H2_Q1_0-0",
            "success": True,
            "status": "ok;legacy_line_scale_used",
            "legacy_policy": "line_scale_used",
            "sigma_nm": 0.027,
            "sigma_lower_bound_nm": 0.015,
            "sigma_upper_bound_nm": 0.0423,
        },
        {
            "line_id": "H2_Q6_1-1",
            "success": True,
            "status": "ok;decontaminated",
            "legacy_policy": "",
            "sigma_nm": 0.029,
            "sigma_lower_bound_nm": 0.015,
            "sigma_upper_bound_nm": 0.0423,
        },
        {
            "line_id": "H2_Q3_0-0",
            "success": True,
            "status": "ok;decontaminated;legacy_deblend_reject",
            "legacy_policy": "deblend_reject",
            "sigma_nm": 0.028,
            "sigma_lower_bound_nm": 0.015,
            "sigma_upper_bound_nm": 0.0423,
        },
        {
            "line_id": "H2_Q4_1-1",
            "success": True,
            "status": "ok;decontaminated;suspicious_decontamination",
            "legacy_policy": "",
            "sigma_nm": 0.030,
            "sigma_lower_bound_nm": 0.015,
            "sigma_upper_bound_nm": 0.0423,
        },
        {
            "line_id": "H2_Q5_0-0",
            "success": True,
            "status": "ok;sigma_at_upper_bound",
            "legacy_policy": "line_scale_used",
            "sigma_nm": 0.0423,
            "sigma_lower_bound_nm": 0.015,
            "sigma_upper_bound_nm": 0.0423,
        },
        {
            "line_id": "H2_Q7_0-0",
            "success": False,
            "status": "fit_failed:RuntimeError",
            "legacy_policy": "deblend_accept",
            "sigma_nm": 0.027,
            "sigma_lower_bound_nm": 0.015,
            "sigma_upper_bound_nm": 0.0423,
        },
    ]
    table = pd.DataFrame(rows)

    mask = clean_sigma_mask(
        table,
        line_ids={"H2_Q1_0-0", "H2_Q6_1-1", "H2_Q3_0-0", "H2_Q4_1-1", "H2_Q5_0-0"},
    )
    kept = table.loc[mask, "line_id"].tolist()
    stats = summarize_sigma(table, mask)

    assert kept == ["H2_Q1_0-0", "H2_Q6_1-1"]
    assert stats.count == 2
    assert stats.min == 0.027
    assert stats.max == 0.029


def test_q7_00_decontamination_keeps_target_area_for_matrix_export():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.output import results_to_matrices
    from fulcher_extractor.spectrocube_io import Spectrum

    sigma = 0.0273
    target = FulcherLine("H2", "Q", 0, 0, "0-0", 7, 609.0374, "synthetic", "nm")
    wavelength = np.linspace(608.80, 609.25, 451)
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.070, target.wavelength_nm - 0.002, sigma)
    intensity += gaussian_area_model(wavelength, 0.017, target.wavelength_nm - 0.055, sigma)
    intensity += gaussian_area_model(wavelength, 0.014, target.wavelength_nm + 0.060, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    result = extract_lines(
        spectrum,
        lines=[target],
        wavelength_min_nm=608.0,
        wavelength_max_nm=610.0,
        config=FitConfig(
            instrument_sigma_nm=sigma,
            instrument_sigma_leeway_nm=0.005,
            close_neighbor_threshold_nm=0.10,
        ),
    )[0]
    intensity_matrix, _ = results_to_matrices([result])

    assert result.success
    assert "decontaminated" in result.status
    assert "legacy_deblend_accept" in result.status
    assert result.contaminant_component_count == 2
    assert result.blend_component_count == 3
    assert abs(result.amplitude - 0.070) < 0.004
    assert result.matrix_amplitude == result.amplitude
    assert intensity_matrix.loc[7, "0-0"] == result.amplitude
    assert [float(value) for value in result.contaminant_amplitudes.split(",")] == sorted(
        [float(value) for value in result.contaminant_amplitudes.split(",")],
        reverse=True,
    )


def test_q3_00_decontamination_preserves_legacy_zeroed_export():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.output import results_to_matrices
    from fulcher_extractor.spectrocube_io import Spectrum

    sigma = 0.0273
    target = FulcherLine("H2", "Q", 0, 0, "0-0", 3, 603.1909, "synthetic", "nm")
    wavelength = np.linspace(603.04, 603.30, 261)
    target_center = target.wavelength_nm - 0.004
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.090, target_center, sigma)
    intensity += gaussian_area_model(wavelength, 0.018, target_center - 0.045, sigma)
    intensity += gaussian_area_model(wavelength, 0.014, target_center + 0.060, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    result = extract_lines(
        spectrum,
        lines=[target],
        wavelength_min_nm=603.0,
        wavelength_max_nm=603.4,
        config=FitConfig(
            instrument_sigma_nm=sigma,
            instrument_sigma_leeway_nm=0.005,
            close_neighbor_threshold_nm=0.10,
        ),
    )[0]
    intensity_matrix, _ = results_to_matrices([result])
    contaminant_centers = [
        float(value) for value in result.contaminant_centers_nm.split(",")
    ]
    contaminant_sigmas = [
        float(value) for value in result.contaminant_sigmas_nm.split(",")
    ]

    assert result.success
    assert "decontaminated" in result.status
    assert "legacy_deblend_reject" in result.status
    assert result.legacy_matrix_action == "zero"
    assert result.contaminant_component_count == 2
    assert result.contaminant_labels == "blue_contaminant,red_shoulder"
    assert abs(result.amplitude - 0.090) < 0.004
    shoulder_offsets = [center - result.center_nm for center in contaminant_centers]
    assert -0.060 <= shoulder_offsets[0] <= -0.030
    assert 0.030 <= shoulder_offsets[1] <= 0.100
    assert all(abs(sigma_nm - result.sigma_nm) < 1e-10 for sigma_nm in contaminant_sigmas)
    assert result.matrix_amplitude == 0.0
    assert intensity_matrix.loc[3, "0-0"] == 0.0


def test_q4_11_decontamination_uses_soft_red_shoulder_distance():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.output import results_to_matrices
    from fulcher_extractor.spectrocube_io import Spectrum

    sigma = 0.0273
    target = FulcherLine("H2", "Q", 1, 1, "1-1", 4, 614.6186, "synthetic", "nm")
    wavelength = np.linspace(614.50, 614.78, 281)
    target_center = target.wavelength_nm + 0.012
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.034, target_center, sigma)
    intensity += gaussian_area_model(wavelength, 0.011, target_center + 0.075, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    result = extract_lines(
        spectrum,
        lines=[target],
        wavelength_min_nm=614.0,
        wavelength_max_nm=615.0,
        config=FitConfig(
            instrument_sigma_nm=sigma,
            instrument_sigma_leeway_nm=0.005,
            close_neighbor_threshold_nm=0.10,
        ),
    )[0]
    intensity_matrix, _ = results_to_matrices([result])
    contaminant_center = float(result.contaminant_centers_nm)

    assert result.success
    assert "decontaminated" in result.status
    assert "suspicious_decontamination" in result.status
    assert "sigma_at_upper_bound" not in result.status
    assert result.contaminant_component_count == 1
    assert result.contaminant_labels == "red_shoulder_neighbor"
    assert result.blend_component_count == 2
    assert 0.0 < result.amplitude < 0.034 + 0.011
    shoulder_offset = contaminant_center - result.center_nm
    assert 0.065 <= shoulder_offset <= 0.085
    assert abs(result.sigma_nm - sigma) < 0.002
    assert result.matrix_amplitude == result.amplitude
    assert intensity_matrix.loc[4, "1-1"] == result.amplitude


def test_q6_11_decontamination_uses_blue_and_red_neighbours():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.output import results_to_matrices
    from fulcher_extractor.spectrocube_io import Spectrum

    sigma = 0.0273
    target = FulcherLine("H2", "Q", 1, 1, "1-1", 6, 617.5462, "synthetic", "nm")
    wavelength = np.linspace(617.25, 617.80, 551)
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.032, target.wavelength_nm + 0.010, sigma)
    intensity += gaussian_area_model(wavelength, 0.018, target.wavelength_nm - 0.145, sigma)
    intensity += gaussian_area_model(wavelength, 0.021, target.wavelength_nm - 0.077, sigma)
    intensity += gaussian_area_model(wavelength, 0.017, target.wavelength_nm + 0.097, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    result = extract_lines(
        spectrum,
        lines=[target],
        wavelength_min_nm=617.0,
        wavelength_max_nm=618.0,
        config=FitConfig(
            instrument_sigma_nm=sigma,
            instrument_sigma_leeway_nm=0.005,
            close_neighbor_threshold_nm=0.10,
        ),
    )[0]
    intensity_matrix, _ = results_to_matrices([result])

    assert result.success
    assert "decontaminated" in result.status
    assert "sigma_at_upper_bound" not in result.status
    assert result.contaminant_component_count == 3
    assert result.contaminant_labels == "blue_neighbor_1,blue_neighbor_2,red_neighbor"
    assert result.blend_component_count == 4
    assert abs(result.amplitude - 0.032) < 0.004
    assert abs(result.sigma_nm - sigma) < 0.002
    assert result.matrix_amplitude == result.amplitude
    assert intensity_matrix.loc[6, "1-1"] == result.amplitude


def test_q2_22_decontamination_uses_soft_blue_shoulder_distance():
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.output import results_to_matrices
    from fulcher_extractor.spectrocube_io import Spectrum

    sigma = 0.0273
    target = FulcherLine("H2", "Q", 2, 2, "2-2", 2, 623.0258, "synthetic", "nm")
    wavelength = np.linspace(622.86, 623.15, 291)
    target_center = target.wavelength_nm - 0.004
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.018, target_center, sigma)
    intensity += gaussian_area_model(wavelength, 0.005, target_center - 0.070, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )

    result = extract_lines(
        spectrum,
        lines=[target],
        wavelength_min_nm=622.8,
        wavelength_max_nm=623.2,
        config=FitConfig(
            instrument_sigma_nm=sigma,
            instrument_sigma_leeway_nm=0.005,
            close_neighbor_threshold_nm=0.10,
        ),
    )[0]
    intensity_matrix, _ = results_to_matrices([result])
    contaminant_center = float(result.contaminant_centers_nm)

    assert result.success
    assert "decontaminated" in result.status
    assert "sigma_at_upper_bound" not in result.status
    assert result.contaminant_component_count == 1
    assert result.contaminant_labels == "blue_shoulder"
    assert abs(result.amplitude - 0.018) < 0.003
    shoulder_offset = contaminant_center - result.center_nm
    assert -0.085 <= shoulder_offset <= -0.055
    assert abs(result.sigma_nm - sigma) < 0.002
    assert result.matrix_amplitude == result.amplitude
    assert intensity_matrix.loc[2, "2-2"] == result.amplitude


def test_archaeology_backed_output_matrix_convention():
    from fulcher_extractor.fit import LineFitResult
    from fulcher_extractor.output import results_to_matrices

    # Historical CMOS H2 table int_152478_f10.txt records these values with
    # convention inte[NN-1][vv], where columns are diagonal bands.
    results = [
        LineFitResult(
            line_id="H2_Q1_0-0",
            isotopologue="H2",
            branch="Q",
            band="0-0",
            N=1,
            rest_wavelength_nm=601.8299,
            amplitude=0.1470036,
            amplitude_stderr=0.0048913,
            center_nm=601.8299,
            center_stderr_nm=0.0,
            sigma_nm=0.05,
            sigma_stderr_nm=0.0,
            fwhm_nm=0.1177,
            baseline_offset=0.0,
            baseline_strategy="local_minimum",
            window_min_nm=601.6,
            window_max_nm=602.0,
            background_min_nm=600.3,
            background_max_nm=603.3,
            n_points=20,
            residual_rms=0.0,
            success=True,
            status="ok",
        ),
        LineFitResult(
            line_id="H2_Q7_1-1",
            isotopologue="H2",
            branch="Q",
            band="1-1",
            N=7,
            rest_wavelength_nm=619.3812,
            amplitude=0.0913015,
            amplitude_stderr=0.0020390,
            center_nm=619.3812,
            center_stderr_nm=0.0,
            sigma_nm=0.05,
            sigma_stderr_nm=0.0,
            fwhm_nm=0.1177,
            baseline_offset=0.0,
            baseline_strategy="local_minimum",
            window_min_nm=619.2,
            window_max_nm=619.6,
            background_min_nm=617.9,
            background_max_nm=620.9,
            n_points=20,
            residual_rms=0.0,
            success=True,
            status="ok",
        ),
    ]

    intensity, error = results_to_matrices(results)

    assert intensity.loc[1, "0-0"] == 0.1470036
    assert error.loc[1, "0-0"] == 0.0048913
    assert intensity.loc[7, "1-1"] == 0.0913015
    assert intensity.loc[1, "1-1"] == 0.0


def test_qc_region_plot_renders_with_band_rails(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    from fulcher_extractor.line_database import load_lines
    from fulcher_extractor.qc import plot_region
    from fulcher_extractor.spectrocube_io import Spectrum

    wavelength = np.linspace(600.0, 630.0, 1201)
    intensity = 0.1 + 0.03 * np.sin(wavelength * 8.0)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 12},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    output = tmp_path / "region.png"
    lines = load_lines()
    label_lines = [
        next(line for line in lines if line.line_id == "H2_Q4_0-0"),
        next(line for line in lines if line.line_id == "H2_Q9_1-1"),
        next(line for line in lines if line.line_id == "H2_Q3_2-2"),
    ]

    fig = plot_region(
        spectrum,
        lines=lines,
        label_lines=label_lines,
        guide_lines=label_lines,
        output_path=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert fig.axes[0].lines
    labels = {text.get_text() for text in fig.axes[0].texts}
    assert {"Q4", "Q9", "Q3"}.issubset(labels)
    assert "Q1" not in labels
    assert len(fig.axes[0].lines) >= 1 + len(label_lines)
    plt.close(fig)


def test_qc_region_plot_can_label_all_lines_for_identification(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    from fulcher_extractor.line_database import load_lines
    from fulcher_extractor.qc import plot_region
    from fulcher_extractor.spectrocube_io import Spectrum

    wavelength = np.linspace(600.0, 630.0, 1201)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 12},
        wavelength_nm=wavelength,
        intensity=np.ones_like(wavelength),
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    lines = load_lines()
    output = tmp_path / "region_all_labels.png"

    fig = plot_region(
        spectrum,
        lines=lines,
        show_all_line_labels=True,
        output_path=output,
    )

    labels = [text.get_text() for text in fig.axes[0].texts]
    assert output.exists()
    assert labels.count("Q1") >= 2
    assert "Q11" in labels
    plt.close(fig)


def test_qc_line_fit_plot_renders_isolated_and_blend(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    from fulcher_extractor.fit import FitConfig, fit_line_group, fit_single_line
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.qc import plot_line_fit
    from fulcher_extractor.spectrocube_io import Spectrum

    sigma = 0.0273
    line_a = FulcherLine("H2", "Q", 1, 1, "1-1", 9, 623.7457, "synthetic", "nm")
    line_b = FulcherLine("H2", "Q", 2, 2, "2-2", 3, 623.8391, "synthetic", "nm")
    wavelength = np.linspace(623.5, 624.05, 551)
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.08, line_a.wavelength_nm, sigma)
    intensity += gaussian_area_model(wavelength, 0.03, line_b.wavelength_nm, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    config = FitConfig(
        instrument_sigma_nm=sigma,
        instrument_sigma_leeway_nm=0.005,
    )

    isolated = fit_single_line(spectrum, line_a, config=config)
    isolated_output = tmp_path / "isolated.png"
    isolated_fig = plot_line_fit(spectrum, isolated, output_path=isolated_output)

    blend_results = fit_line_group(spectrum, [line_a, line_b], config=config)
    components = [
        {
            "label": result.line_id,
            "amplitude": result.amplitude,
            "center_nm": result.center_nm,
            "sigma_nm": result.sigma_nm,
            "rest_wavelength_nm": result.rest_wavelength_nm,
        }
        for result in blend_results
    ]
    blend_output = tmp_path / "blend.png"
    blend_fig = plot_line_fit(
        spectrum,
        blend_results[0],
        components=components,
        neighbor_lines=[line_a, line_b],
        output_path=blend_output,
    )

    assert isolated_output.exists()
    assert isolated_output.stat().st_size > 0
    assert any(collection.get_hatch() for collection in isolated_fig.axes[0].collections)
    assert blend_output.exists()
    assert blend_output.stat().st_size > 0
    assert any(collection.get_hatch() for collection in blend_fig.axes[0].collections)
    assert blend_fig.axes[0].get_title(loc="left") == "H2 Q9(1-1) + Q3(2-2) fit QC"
    assert not any("fit/rest" in text.get_text() for ax in blend_fig.axes for text in ax.texts)
    assert any("target:" in text.get_text() for text in blend_fig.axes[1].texts)
    plt.close(isolated_fig)
    plt.close(blend_fig)


def test_qc_line_fit_plot_labels_unresolved_coincident_blend(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    from fulcher_extractor.fit import LineFitResult
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.qc import plot_line_fit
    from fulcher_extractor.spectrocube_io import Spectrum

    wavelength = np.linspace(626.05, 626.45, 401)
    sigma = 0.0273
    center = 626.2495
    intensity = 1.0 + gaussian_area_model(wavelength, 0.08, center, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    result = LineFitResult(
        line_id="H2_Q10_1-1",
        isotopologue="H2",
        branch="Q",
        band="1-1",
        N=10,
        rest_wavelength_nm=center,
        amplitude=0.05,
        amplitude_stderr=0.001,
        center_nm=center,
        center_stderr_nm=0.0,
        sigma_nm=sigma,
        sigma_stderr_nm=0.0,
        fwhm_nm=2.354820045 * sigma,
        baseline_offset=1.0,
        baseline_strategy="local_minimum",
        window_min_nm=626.05,
        window_max_nm=626.45,
        background_min_nm=625.0,
        background_max_nm=627.5,
        n_points=wavelength.size,
        residual_rms=0.0,
        success=True,
        status="ok;blend_group;unresolved_coincident_database_lines",
        expected_center_nm=center,
        center_offset_from_rest_nm=0.0,
        center_offset_from_expected_nm=0.0,
        sigma_lower_bound_nm=0.015,
        sigma_upper_bound_nm=0.0423,
        center_lower_bound_nm=626.1,
        center_upper_bound_nm=626.33,
        blend_group_id="H2_Q10_1-1+H2_Q5_2-2",
        blend_component_count=2,
        close_neighbor_ids="H2_Q10_1-1,H2_Q5_2-2",
        blend_delta_nm=0.0,
    )
    components = [
        {
            "label": "H2_Q10_1-1",
            "amplitude": 0.05,
            "center_nm": center,
            "sigma_nm": sigma,
            "rest_wavelength_nm": center,
        },
        {
            "label": "H2_Q5_2-2",
            "amplitude": 0.03,
            "center_nm": center,
            "sigma_nm": sigma,
            "rest_wavelength_nm": center,
        },
    ]
    output = tmp_path / "coincident.png"

    fig = plot_line_fit(spectrum, result, components=components, output_path=output)

    assert output.exists()
    assert output.stat().st_size > 0
    assert any(
        "unresolved: no spectral split" in text.get_text()
        for text in fig.axes[1].texts
    )
    assert fig.axes[1].get_title(loc="left").startswith("status: ok;blend_group")
    plt.close(fig)


def test_qc_line_fit_plot_can_hide_markers_and_show_contamination(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    from fulcher_extractor.fit import FitConfig, fit_single_line
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.qc import plot_line_fit
    from fulcher_extractor.spectrocube_io import Spectrum

    sigma = 0.0273
    target = FulcherLine("H2", "Q", 1, 1, "1-1", 4, 608.55, "synthetic", "nm")
    wavelength = np.linspace(608.35, 608.85, 501)
    contaminant_center = 608.64
    intensity = 1.0
    intensity += gaussian_area_model(wavelength, 0.06, target.wavelength_nm, sigma)
    intensity += gaussian_area_model(wavelength, 0.025, contaminant_center, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    result = fit_single_line(
        spectrum,
        target,
        config=FitConfig(instrument_sigma_nm=sigma, instrument_sigma_leeway_nm=0.005),
    )
    components = [
        {
            "label": result.line_id,
            "amplitude": result.amplitude,
            "center_nm": result.center_nm,
            "sigma_nm": result.sigma_nm,
            "rest_wavelength_nm": result.rest_wavelength_nm,
        },
        {
            "label": "contamination",
            "display_label": "blended",
            "role": "contaminant",
            "amplitude": 0.025,
            "center_nm": contaminant_center,
            "sigma_nm": sigma,
        },
    ]
    output = tmp_path / "contamination.png"

    fig = plot_line_fit(
        spectrum,
        result,
        components=components,
        show_database_markers=False,
        show_fitted_markers=False,
        output_path=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert fig.axes[0].get_title(loc="left") == "H2 Q4(1-1) fit QC"
    assert any("blended" == text.get_text() for text in fig.axes[0].texts)
    assert not any(text.get_text() in {"db", "fit"} for text in fig.axes[0].texts)
    assert sum(bool(collection.get_hatch()) for collection in fig.axes[0].collections) >= 2
    assert "status:" not in "\n".join(text.get_text() for text in fig.axes[1].texts)
    assert fig.axes[1].get_title(loc="left").startswith("status: ok")
    plt.close(fig)


def test_qc_line_fit_page_pdf_and_optional_pngs(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.qc import plot_line_fit_page, write_line_fit_qc
    from fulcher_extractor.spectrocube_io import Spectrum

    sigma = 0.0273
    lines = [
        FulcherLine("H2", "Q", 0, 0, "0-0", 4, 607.4827, "synthetic", "nm"),
        FulcherLine("H2", "Q", 1, 1, "1-1", 9, 623.7457, "synthetic", "nm"),
        FulcherLine("H2", "Q", 2, 2, "2-2", 3, 623.8391, "synthetic", "nm"),
        FulcherLine("H2", "Q", 1, 1, "1-1", 10, 626.2495, "synthetic", "nm"),
        FulcherLine("H2", "Q", 2, 2, "2-2", 5, 626.2495, "synthetic", "nm"),
    ]
    wavelength = np.linspace(606.9, 626.7, 2200)
    intensity = 1.0 + 0.01 * np.sin(wavelength * 3.0)
    for line, area in zip(lines, [0.12, 0.08, 0.03, 0.05, 0.025]):
        intensity += gaussian_area_model(wavelength, area, line.wavelength_nm, sigma)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    results = extract_lines(
        spectrum,
        lines=lines,
        wavelength_min_nm=606.0,
        wavelength_max_nm=627.0,
        config=FitConfig(
            instrument_sigma_nm=sigma,
            instrument_sigma_leeway_nm=0.005,
            close_neighbor_threshold_nm=0.10,
        ),
    )
    pdf_output = tmp_path / "frame_line_fits.pdf"

    fig = plot_line_fit_page(spectrum, results, columns=2, output_path=pdf_output)

    assert pdf_output.exists()
    assert pdf_output.stat().st_size > 0
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 3
    assert visible_axes[0].get_title(loc="left") == "Q4(0-0)"
    assert visible_axes[1].get_title(loc="left") == "Q9(1-1) + Q3(2-2)"
    assert not visible_axes[2].get_title(loc="right")
    assert any("unresolved" in text.get_text() for text in visible_axes[2].texts)
    x_limits = [ax.get_xlim() for ax in visible_axes]
    x_widths = [right - left for left, right in x_limits]
    assert np.allclose(x_widths, x_widths[0])
    by_id = {result.line_id: result for result in results}
    expected_centers = [
        by_id["H2_Q4_0-0"].center_nm,
        0.5
        * (
            min(by_id["H2_Q9_1-1"].center_nm, by_id["H2_Q3_2-2"].center_nm)
            + max(by_id["H2_Q9_1-1"].center_nm, by_id["H2_Q3_2-2"].center_nm)
        ),
        0.5
        * (
            min(by_id["H2_Q10_1-1"].center_nm, by_id["H2_Q5_2-2"].center_nm)
            + max(by_id["H2_Q10_1-1"].center_nm, by_id["H2_Q5_2-2"].center_nm)
        ),
    ]
    actual_centers = [0.5 * (left + right) for left, right in x_limits]
    assert np.allclose(actual_centers, expected_centers, atol=1e-9)
    wavelength_step = wavelength[1] - wavelength[0]
    for ax, (left, right) in zip(visible_axes, x_limits):
        data_x = ax.lines[0].get_xdata()
        assert data_x.min() <= left + wavelength_step
        assert data_x.max() >= right - wavelength_step
    plt.close(fig)

    open_figures_before = set(plt.get_fignums())
    written = write_line_fit_qc(
        spectrum,
        results,
        pdf_path=tmp_path / "frame_line_fits_wrapped.pdf",
        individual_dir=tmp_path / "line_pngs",
        save_individual_pngs=True,
        columns=2,
    )

    assert len(written) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in written)
    assert len(list((tmp_path / "line_pngs").glob("*.png"))) == 3
    assert set(plt.get_fignums()) == open_figures_before


def test_qc_line_fit_pages_can_append_to_one_pdf(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.qc import plot_line_fit_page
    from fulcher_extractor.spectrocube_io import Spectrum

    line = FulcherLine("H2", "Q", 0, 0, "0-0", 4, 607.4827, "synthetic", "nm")
    wavelength = np.linspace(607.0, 608.0, 500)
    pdf_output = tmp_path / "cube_line_fits.pdf"

    with PdfPages(pdf_output) as pdf:
        for frame in (0, 1):
            intensity = 1.0 + gaussian_area_model(wavelength, 0.10 + 0.01 * frame, line.wavelength_nm, 0.0273)
            spectrum = Spectrum(
                source_path=__file__,
                shot_id="synthetic",
                selectors={"frame": frame},
                wavelength_nm=wavelength,
                intensity=intensity,
                intensity_units="a.u.",
                wavelength_medium="air",
                metadata={},
            )
            results = extract_lines(
                spectrum,
                lines=[line],
                wavelength_min_nm=607.0,
                wavelength_max_nm=608.0,
                config=FitConfig(instrument_sigma_nm=0.0273),
            )
            fig = plot_line_fit_page(spectrum, results, columns=1)
            pdf.savefig(fig)
            plt.close(fig)

    assert pdf_output.exists()
    assert pdf_output.stat().st_size > 0
    assert not plt.get_fignums()


def test_h2_plot_command_reloads_fit_report_results(tmp_path):
    import numpy as np

    from fulcher_extractor.extract import extract_lines
    from fulcher_extractor.fit import FitConfig
    from fulcher_extractor.h2_dataset_cli import _line_fit_results_from_csv
    from fulcher_extractor.line_database import FulcherLine
    from fulcher_extractor.line_models import gaussian_area_model
    from fulcher_extractor.output import results_to_dataframe
    from fulcher_extractor.spectrocube_io import Spectrum

    line = FulcherLine("H2", "Q", 0, 0, "0-0", 4, 607.4827, "synthetic", "nm")
    wavelength = np.linspace(607.0, 608.0, 500)
    intensity = 1.0 + gaussian_area_model(wavelength, 0.10, line.wavelength_nm, 0.0273)
    spectrum = Spectrum(
        source_path=__file__,
        shot_id="synthetic",
        selectors={"frame": 0},
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_units="a.u.",
        wavelength_medium="air",
        metadata={},
    )
    results = extract_lines(
        spectrum,
        lines=[line],
        wavelength_min_nm=607.0,
        wavelength_max_nm=608.0,
        config=FitConfig(instrument_sigma_nm=0.0273),
    )
    report_path = tmp_path / "fit_report.csv"
    results_to_dataframe(results).to_csv(report_path, index=False)

    reloaded = _line_fit_results_from_csv(report_path)

    assert len(reloaded) == 1
    assert reloaded[0].line_id == results[0].line_id
    assert reloaded[0].N == results[0].N
    assert reloaded[0].success is True
    assert np.isclose(reloaded[0].amplitude, results[0].amplitude)


def test_plot_plan_inherits_extract_section(tmp_path):
    import argparse

    from fulcher_extractor.h2_dataset_cli import _apply_plan

    plan_path = tmp_path / "h2_dataset_plan.toml"
    plan_path.write_text(
        "\n".join(
            [
                "[common]",
                'engine = "h5netcdf"',
                "",
                "[extract]",
                'output_dir = "dataset"',
                'manifest = "scan/selected_frames.csv"',
                "qc_every = 1",
                "line_fit_qc = true",
                "workers = 4",
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="plot",
        engine=None,
        output_dir=None,
        manifest=None,
        qc_every=0,
        line_fit_qc=False,
        workers=1,
    )

    _apply_plan(args, plan_path, provided=set())

    assert args.engine == "h5netcdf"
    assert args.output_dir == tmp_path / "dataset"
    assert args.manifest == tmp_path / "scan" / "selected_frames.csv"
    assert args.qc_every == 1
    assert args.line_fit_qc is True
    assert args.workers == 4


def test_extraction_progress_resume_requires_complete_artifacts(tmp_path):
    from fulcher_extractor.h2_dataset_cli import (
        _append_extraction_progress,
        _frame_key,
        _load_extraction_progress,
    )

    output_dir = tmp_path / "dataset"
    result = {
        "row": {
            "shot": "193790",
            "frame": 15,
            "cube": "C:/data/193790_spectrocube.nc",
            "status": "ok",
            "n_lines": 29,
        },
        "policy_rows": [
            {
                "shot": "193790",
                "frame": 15,
                "legacy_policy": "accepted",
                "legacy_matrix_action": "keep",
                "n_lines": 29,
            }
        ],
    }
    progress_path = output_dir / "extraction_progress.jsonl"

    _append_extraction_progress(progress_path, result)

    assert _load_extraction_progress(progress_path, output_dir) == {}

    for relative in [
        "intensities/193790_fr_15.csv",
        "intensities/193790_fr_15_err.csv",
        "fit_reports/193790_fr_15_fit_report.csv",
    ]:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")

    loaded = _load_extraction_progress(progress_path, output_dir)

    assert list(loaded) == [_frame_key(result["row"])]
    assert loaded[_frame_key(result["row"])]["row"]["n_lines"] == 29
