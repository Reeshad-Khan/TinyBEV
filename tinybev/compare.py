"""Compare measured validation results with TinyBEV Table 2."""
import argparse
import json
import math
from pathlib import Path

METRICS = ("map", "nds", "amota", "amotp", "l2_3s", "minade", "collision_pct", "fps")
TABLE_2 = {
    "S0": (31.0, 30.0, 28.0, 2.00, 1.43, 1.00, .48, 11),
    "S1": (38.0, 32.1, 30.0, 1.75, 1.40, .85, .45, 11),
    "S2": (31.0, 30.5, 29.0, 1.90, 1.22, .82, .39, 11),
    "S3": (39.0, 34.2, 34.0, 1.60, 1.08, .78, .32, 11),
}
PRECISION = (1, 1, 1, 2, 2, 2, 2, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="Exit 1 on any missing or mismatched measurement")
    parser.add_argument("--rank", action="store_true",
                        help="Rank complete runs per variant by mean absolute relative error")
    args = parser.parse_args()
    success = True
    candidates = {variant: [] for variant in TABLE_2}
    for path in args.metrics:
        try:
            measured = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(f"{path}: {error}")
        if not isinstance(measured, dict) or measured.get("split") != "val":
            parser.error(f"{path}: Table 2 uses the full val split")
        variant = measured.get("variant")
        if not isinstance(variant, str) or variant not in TABLE_2:
            parser.error(f"{path}: unknown student variant {variant!r}")
        errors = []
        print(f"{variant} ({path})")
        print(f"{'metric':16} {'paper':>8} {'observed':>10} {'delta':>10}")
        for key, target, digits in zip(METRICS, TABLE_2[variant], PRECISION):
            value = measured.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                print(f"{key:16} {target:8.{digits}f} {'missing':>10}")
                success = False
                continue
            print(f"{key:16} {target:8.{digits}f} {value:10.{digits + 2}f} {value-target:+10.{digits + 2}f}")
            success &= round(value, digits) == target
            errors.append(abs(value - target) / target)
        if len(errors) == len(METRICS):
            candidates[variant].append((100 * sum(errors) / len(errors), str(path)))
    if args.rank:
        print("\nMean absolute relative error across Table 2 metrics (lower is closer)")
        for variant, runs in candidates.items():
            for error, path in sorted(runs):
                print(f"{variant}  {error:8.3f}%  {path}")
    if args.check and not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
