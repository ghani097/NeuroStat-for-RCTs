import argparse
import sys
from pathlib import Path
from pli_pipeline import run_study_pipeline, preprocess_file, find_set_files, StudyDesign, PipelineConfig


def progress_cb(msg: str) -> None:
    print(msg, flush=True)


def _apply_cleaning_method(cfg: PipelineConfig, cleaning_method: str) -> None:
    if cleaning_method == "gedai":
        cfg.preprocessing.use_ged = True
        cfg.preprocessing.use_asr = False
        cfg.preprocessing.use_iclabel = False
        print("Cleaning method: GEDAI", flush=True)
    else:
        cfg.preprocessing.use_ged = False
        cfg.preprocessing.use_asr = True
        cfg.preprocessing.use_iclabel = True
        print("Cleaning method: ASR + ICA", flush=True)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Run NeuroStat EEG pipeline",
    )
    parser.add_argument("base_folder", help="Root folder containing group/session sub-folders")
    parser.add_argument("groups",         nargs="?", default="GroupA,GroupB",
                        help="Comma-separated group names")
    parser.add_argument("sessions",       nargs="?", default="pre,post",
                        help="Comma-separated session names")
    parser.add_argument("output_folder",  nargs="?", default="processed",
                        help="Output root folder")
    parser.add_argument("cleaning_method", nargs="?", default="asr_ica",
                        help="asr_ica | gedai")
    parser.add_argument(
        "--preprocessing-only",
        action="store_true",
        help=(
            "Run preprocessing only — no source localisation, no PLI. "
            "Saves cleaned .fif and .set files for downstream steps."
        ),
    )

    args = parser.parse_args(argv[1:])

    base     = Path(args.base_folder)
    groups   = [g.strip() for g in args.groups.split(",")   if g.strip()]
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    out      = Path(args.output_folder)

    design = StudyDesign(groups=groups, sessions=sessions)

    cfg = PipelineConfig()
    cfg.output_root = out.expanduser()
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    _apply_cleaning_method(cfg, args.cleaning_method)

    print(f"Base folder:  {base}",         flush=True)
    print(f"Groups:       {groups}",        flush=True)
    print(f"Sessions:     {sessions}",      flush=True)
    print(f"Output root:  {cfg.output_root}", flush=True)

    if args.preprocessing_only:
        # ------------------------------------------------------------------
        # Preprocessing-only mode: clean data and save to disk.
        # Source localisation and PLI are NOT run.
        # Each file is processed independently so memory is freed between files.
        # ------------------------------------------------------------------
        print("Mode: PREPROCESSING ONLY — source localisation and PLI skipped.", flush=True)
        summaries = []
        for group in groups:
            for session in sessions:
                session_dir = base / group / session
                if not session_dir.exists():
                    print(f"Warning: session folder not found: {session_dir}", flush=True)
                    continue
                try:
                    set_files = find_set_files(session_dir)
                except FileNotFoundError:
                    print(f"No .set files found in {session_dir}; skipping.", flush=True)
                    continue
                for f in set_files:
                    try:
                        summary = preprocess_file(f, cfg, progress_cb)
                        summaries.append(summary)
                    except Exception as exc:
                        print(f"Failed to preprocess {f.name}: {exc}", flush=True)

        print(f"Finished. Preprocessed {len(summaries)} files.", flush=True)
        print("PREPROCESSING_COMPLETE", flush=True)

    else:
        # ------------------------------------------------------------------
        # Full pipeline: preprocessing + source localisation + PLI.
        # Only reached when ResearchBuddy explicitly needs source loc / PLI.
        # ------------------------------------------------------------------
        print("Mode: FULL PIPELINE — preprocessing + source localisation + PLI.", flush=True)
        summaries, table = run_study_pipeline(base, design, cfg, progress_cb=progress_cb)
        print(f"Finished. Processed {len(summaries)} files.", flush=True)
        print(f"Study table saved to: {cfg.output_root / 'PLI_Table.xlsx'}", flush=True)
        print("PREPROCESSING_COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv)
