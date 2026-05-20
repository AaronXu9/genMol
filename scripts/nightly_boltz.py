"""Re-score the latest top-K generated molecules with Boltz-2. M4+."""
from __future__ import annotations


def main():
    raise NotImplementedError(
        "M4: rank latest samples by Vina, pick top-K, send to BoltzOracle subprocess,"
        " write reports/nightly_boltz/<date>.md"
    )


if __name__ == "__main__":
    main()
