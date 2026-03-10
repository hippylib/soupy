This directory stores outputs from
`driver_poisson_compare_models_fixed_controls.py`.

The comparison is done at fixed controls `z` (no optimization), including:
- zero / affine / sinusoidal controls with different amplitudes
- random controls with different amplitudes

Expected outputs:
- `fixed_control_model_errors.csv`
- `fixed_control_model_summary.csv`
- `fixed_control_truth_costs.json`
- `fixed_control_abs_error_comparison.png`
