import sys
from pathlib import Path
from pli_pipeline import run_study_pipeline, StudyDesign, PipelineConfig

def progress_cb(msg: str) -> None:
    print(msg, flush=True)

def main(argv):
    if len(argv) < 2:
        print("Usage: python run_study.py <base_folder> [groups_csv] [sessions_csv] [output_folder]")
        return

    base = Path(argv[1])
    groups = (argv[2].split(",") if len(argv) > 2 else ["GroupA","GroupB"])
    sessions = (argv[3].split(",") if len(argv) > 3 else ["pre","post"])
    out = Path(argv[4]) if len(argv) > 4 else (Path.cwd() / "processed")

    design = StudyDesign(groups=[g.strip() for g in groups if g.strip()],
                         sessions=[s.strip() for s in sessions if s.strip()])

    cfg = PipelineConfig()
    cfg.output_root = out.expanduser()
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Base folder: {base}")
    print(f"Groups: {design.groups}")
    print(f"Sessions: {design.sessions}")
    print(f"Output root: {cfg.output_root}")

    summaries, table = run_study_pipeline(base, design, cfg, progress_cb=progress_cb)
    print(f"Finished. Processed {len(summaries)} files.")
    print(f"Study table saved to: {cfg.output_root / 'PLI_Table.xlsx'}")

if __name__ == "__main__":
    main(sys.argv)