"""Repository paths, in one place (modules must not locate data relative to their own file)."""

from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
RESULTS = ROOT / "results"          # trial results and scheduler start records (gitignored)
DB = ROOT / "db"                    # the old SQLite files (kept as a backup after the move to Postgres)
WEB = PKG / "web"                   # pages and assets served by the API and the analysis dashboard
ENGINE = PKG / "engine"             # files uploaded into GPU sandboxes (stdlib only)
PATCHES = PKG / "patches"           # scripts run at image build time
EXPERIMENTS = ROOT / "experiments"  # one-off probes and earlier experiments
