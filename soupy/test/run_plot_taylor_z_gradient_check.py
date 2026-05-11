"""Batch runner for plot_taylor_z_gradient_check.py.

Runs the Taylor gradient-check plotting script for a fixed list of parameter
tuples, stores generated figures under a dedicated output directory, and
records terminal output both per case and in a combined summary file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PLOT_SCRIPT = THIS_DIR / "plot_taylor_z_gradient_check.py"


CASES = [
    (1.0, 5.0, 0.0),
    (0.2, 1.0, 0.0),
    (0.1, 0.5, 0.0),
]


def _format_float(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def _case_name(gamma: float, delta: float, z_value: float) -> str:
    return (
        f"gamma_{_format_float(gamma)}"
        f"_delta_{_format_float(delta)}"
        f"_z_{_format_float(z_value)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run plot_taylor_z_gradient_check.py over a fixed batch of parameters."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(THIS_DIR / "plot_taylor_z_gradient_check_batch_outputs"),
        help="Directory where per-case images and logs will be stored.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Variance beta passed to the plot script.",
    )
    parser.add_argument(
        "--n-mix",
        type=int,
        default=39,
        help="N_mix passed to the plot script.",
    )
    parser.add_argument(
        "--n-tr",
        type=int,
        default=50,
        help="N_tr passed to the plot script.",
    )
    parser.add_argument("--nx", type=int, default=20, help="Mesh cells in x.")
    parser.add_argument("--ny", type=int, default=20, help="Mesh cells in y.")
    parser.add_argument("--eps-min", type=float, default=1e-4, help="Minimum epsilon.")
    parser.add_argument("--eps-max", type=float, default=1e2, help="Maximum epsilon.")
    parser.add_argument("--n-eps", type=int, default=16, help="Number of epsilon values.")
    parser.add_argument("--fit-start", type=int, default=3, help="Fit-window start index.")
    parser.add_argument("--fit-end", type=int, default=12, help="Fit-window end index.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dz.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "combined_terminal_output.txt"

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("Batch results for plot_taylor_z_gradient_check.py\n\n")

        for gamma, delta, z_value in CASES:
            case_dir = output_dir / _case_name(gamma, delta, z_value)
            case_dir.mkdir(parents=True, exist_ok=True)

            figure_path = case_dir / "taylor_z_gradient_fd_error.png"
            terminal_log_path = case_dir / "terminal_output.txt"

            header = f"Result for gamma={gamma}, delta={delta}, z={z_value}:"
            print(header)

            cmd = [
                sys.executable,
                str(PLOT_SCRIPT),
                "--gamma",
                str(gamma),
                "--delta",
                str(delta),
                "--z-value",
                str(z_value),
                "--beta",
                str(args.beta),
                "--n-mix",
                str(args.n_mix),
                "--n-tr",
                str(args.n_tr),
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
                "--eps-min",
                str(args.eps_min),
                "--eps-max",
                str(args.eps_max),
                "--n-eps",
                str(args.n_eps),
                "--fit-start",
                str(args.fit_start),
                "--fit-end",
                str(args.fit_end),
                "--seed",
                str(args.seed),
                "--out",
                str(figure_path),
            ]

            completed = subprocess.run(
                cmd,
                cwd=str(THIS_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

            output_text = completed.stdout
            terminal_log_path.write_text(output_text, encoding="utf-8")

            summary_file.write(header + "\n\n")
            summary_file.write(output_text)
            if not output_text.endswith("\n"):
                summary_file.write("\n")
            summary_file.write("\n")
            summary_file.flush()

            if completed.returncode == 0:
                print(f"  completed; logs: {terminal_log_path}")
                print(f"  figures: {case_dir}")
            else:
                print(
                    f"  failed with return code {completed.returncode}; "
                    f"logs: {terminal_log_path}"
                )

    print(f"\nCombined terminal output saved to: {summary_path}")


if __name__ == "__main__":
    main()
