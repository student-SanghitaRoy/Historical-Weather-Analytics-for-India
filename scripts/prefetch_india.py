"""Download and cache daily ERA5 data (1991-2025) for the India locations. Resumable: run it again to continue.

  python scripts/prefetch_india.py --dry-run
  python scripts/prefetch_india.py --max-new 15          # polite batch; repeat later for the rest
  python scripts/prefetch_india.py --cities Guwahati,Delhi
  python scripts/prefetch_india.py --auto                # start once and leave it: waits out hourly limits by itself

The free API has rate limits (they can change; check open-meteo.com). Keep a delay between requests and
download in batches. If a request is refused, the script waits and retries, then moves on.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_platform.data.cities import load_cities  # noqa: E402
from weather_platform.india import (CACHE_DIR, PREFETCH_END, PREFETCH_START, load_datasets, prefetch,  # noqa: E402
                                    prefetch_until_done)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=3.0, help="seconds to wait between downloads")
    ap.add_argument("--max-new", type=int, default=None, help="download at most this many new locations this run")
    ap.add_argument("--cities", default="", help="comma-separated names (default: all)")
    ap.add_argument("--cache-dir", default=None, help="default: data_cache/india next to this project")
    ap.add_argument("--auto", action="store_true", help="keep going until all locations are cached, sleeping through API limits")
    ap.add_argument("--wait-minutes", type=float, default=65.0, help="pause after hitting an hourly limit (with --auto)")
    ap.add_argument("--max-rounds", type=int, default=12, help="give up after this many rounds (with --auto)")
    ap.add_argument("--dry-run", action="store_true", help="only show what is cached and what is missing")
    a = ap.parse_args(argv)

    a.cache_dir = a.cache_dir or CACHE_DIR
    print(f"Data folder: {a.cache_dir}")
    cities = load_cities()
    if a.cities:
        want = {n.strip().lower() for n in a.cities.split(",")}
        cities = [c for c in cities if c["name"].lower() in want]
    _, missing = load_datasets(cities, a.cache_dir)
    print(f"{len(cities) - len(missing)} of {len(cities)} locations cached; {len(missing)} missing ({PREFETCH_START} to {PREFETCH_END}).")
    if a.dry_run:
        print("Missing:", ", ".join(missing) or "none")
        return 0
    if a.auto:
        print("Auto mode: keep this window open and keep the computer awake (Windows sleep pauses the download). Ctrl+C stops safely.")
        try:
            out = prefetch_until_done(cities, a.cache_dir, delay=a.delay, wait_minutes=a.wait_minutes, max_rounds=a.max_rounds)
        except KeyboardInterrupt:
            print("\nStopped. Run the same command again to continue; nothing already downloaded is lost.")
            return 0
        print(f"\nFinished: {'all locations cached' if out['complete'] else str(len(out['remaining'])) + ' left'} after {out['rounds']} round(s).")
        return 0
    res = prefetch(cities, a.cache_dir, delay=a.delay, max_new=a.max_new)
    print(f"\nDownloaded {len(res['fetched'])}, already cached {len(res['cached'])}, failed {len(res['failed'])}, left for later {len(res['skipped'])}.")
    for k, v in res["failed"].items():
        print(f"  FAILED {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
