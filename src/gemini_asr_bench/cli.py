"""Back-compat CLI entry: `python -m gemini_asr_bench.cli` → ndizi_asr.cli."""

from ndizi_asr.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
