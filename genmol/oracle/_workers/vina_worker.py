"""Vina worker — runs inside conda env 'unidock2'.

Reads a JSON payload from stdin: {receptor: str, mols: [paths], exhaustiveness, num_modes}.
Writes JSON to stdout: {results: [{score, metadata?, success, error?}, ...]}.

Implementation deferred to M3 when Vina integration is exercised.
"""
import json
import sys


def main():
    payload = json.loads(sys.stdin.read())
    n = len(payload.get("mols", []))
    out = {
        "results": [
            {
                "score": float("nan"),
                "success": False,
                "error": "M3 stub: implement unidock2 invocation here",
            }
            for _ in range(n)
        ]
    }
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
