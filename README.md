# data-live

Whole SPX and RUT option boards recorded every 15 minutes during US market
hours by `.github/workflows/record.yml` on `main`.

    snapshots/<spx|rut>/<YYYY-MM-DD>/<HHMM>.csv.gz   (New York time)

Format and loading: `docs/recorder.md` on `main`. This branch has no code;
fetch it with `git fetch origin data-live` and read files with
`hypsmile.marketdata.load_snapshot(path, expiry)`.
