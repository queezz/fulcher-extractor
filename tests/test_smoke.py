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
