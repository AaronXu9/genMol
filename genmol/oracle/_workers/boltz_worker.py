"""Boltz-2 worker — runs inside conda env 'boltzina_env'.

Reads {receptor, mols} JSON; writes {results: [{affinity, iptm?, plddt?}, ...]}.
Implementation deferred to M4.
"""
import json
import sys


def main():
    payload = json.loads(sys.stdin.read())
    n = len(payload.get("mols", []))
    out = {
        "results": [
            {
                "affinity": float("nan"),
                "success": False,
                "error": "M4 stub: implement Boltz-2 invocation here",
            }
            for _ in range(n)
        ]
    }
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
