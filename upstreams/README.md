# Bundled runtime sources

These archives contain the upstream code and assets used by game-rsi, exported
without changes from the commits in `../upstreams.lock.json`. The lock file records
each archive's SHA-256 and the included upstream paths. Model weights, generated
data, upstream Git history, demo sites, and unrelated research tooling are omitted.

- `PlayJev.tar.gz`: `playjev/`, all ten games and their shared assets, plus upstream
  licensing and reference files.
- `RSI-Jev.tar.gz`: `rsijev/` and `serve/` (required by `rsijev.Decider`), plus
  upstream licensing, package metadata, and reference files.

Restore sources without accessing upstream Git servers or installing packages:

```bash
python3 scripts/bootstrap.py --sources-only
```

The normal bootstrap command also restores these archives before installing
dependencies. Extracted files live in the ignored `vendor/` directory. Existing
checkouts at the pinned commit are reused. Directories with a different version
are kept intact; move them aside before restoring the bundled version.

To update a snapshot, edit the pinned commit and included paths in the lock file,
prepare clean local Git checkouts at those commits in `vendor/`, and run:

```bash
python3 scripts/bundle_upstreams.py
```

Commit both archives and the updated lock file. See `../THIRD_PARTY.md` and the
licenses inside each archive for attribution and asset licensing.
