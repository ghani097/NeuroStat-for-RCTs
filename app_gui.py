from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:  # pragma: no cover
    tk = None
    filedialog = None
    messagebox = None

# Optional image preview support
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None

from pli_pipeline import (
    LOGGER,
    LOG_FORMAT,
    StudyDesign,
    PipelineConfig,
    run_preprocessing_for_design,
    run_source_and_connectivity_for_design,
    aggregate_csv_table,
    run_stats_for_design,
)


class PLIPipelineApp:
    def __init__(self) -> None:
        if tk is None:
            raise RuntimeError("tkinter is not available in this environment")

        self.root = tk.Tk()
        self.root.title("NeuroStat_Pipeline for RCTs")
        self.root.geometry("1200x720")
        self.root.resizable(True, True)

        # State
        self.base_var = tk.StringVar()
        self.groups_var = tk.StringVar(value="GroupA,GroupB")
        self.sessions_var = tk.StringVar(value="pre,post")
        self.out_var = tk.StringVar(value=str(Path.cwd() / "processed"))

        self.l_freq_var = tk.StringVar(value="1.0")
        self.h_freq_var = tk.StringVar(value="40.0")
        self.notch_var = tk.StringVar(value="50,100")
        self.asr_var = tk.BooleanVar(value=True)
        self.iclabel_var = tk.BooleanVar(value=True)
        self.iclabel_thr_var = tk.StringVar(value="0.85")
        # Stage 3 options
        self.use_gpu_var = tk.BooleanVar(value=True)  # True=auto GPU if available; False=CPU only
        self.n_perm_var = tk.StringVar(value="1000")
        self.cluster_thr_var = tk.StringVar(value="")  # empty => default

        self.preview_img = None  # keep a reference
        self.preview_img_orig = None  # PIL image for dynamic resize
        self.preview_path = None
        self.brain_img = None
        self.brain_img_orig = None
        self.brain_img_label = None

        # Run All step selection flags
        self.run_preprocessing_var = tk.BooleanVar(value=True)
        self.run_source_var = tk.BooleanVar(value=True)
        self.run_csv_var = tk.BooleanVar(value=True)
        self.run_stats_var = tk.BooleanVar(value=True)

        self._build_layout()

    # ---------- UI Builders ----------
    def _build_layout(self) -> None:
        tk.Label(self.root, text="NeuroStat_Pipeline for RCTs: design → preprocessing → source/connectivity/statistics").pack(pady=(6, 4))
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=8, pady=6)

        # Column 1: Study design
        col1 = tk.LabelFrame(container, text="1) Study Design", width=380)
        col1.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=4)
        self._build_col1(col1)

        # Column 2: Preprocessing
        col2 = tk.LabelFrame(container, text="2) Preprocessing", width=420)
        col2.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        self._build_col2(col2)

        # Column 3: Source/Connectivity/Stats
        col3 = tk.LabelFrame(container, text="3) Source / Connectivity / Stats", width=380)
        col3.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        self._build_col3(col3)

        # Bottom panel: Run All
        self._build_run_all_panel()

    def _build_col1(self, frame: tk.Frame) -> None:
        # Base folder
        tk.Label(frame, text="Base folder (contains group folders)").pack(anchor="w", padx=6, pady=(8, 0))
        entry = tk.Entry(frame, textvariable=self.base_var)
        entry.pack(fill="x", padx=6, pady=4)
        tk.Button(frame, text="Browse…", command=self._browse_base).pack(anchor="e", padx=6, pady=(0, 8))

        # Groups & sessions
        tk.Label(frame, text="Groups (comma-separated)").pack(anchor="w", padx=6, pady=(6, 0))
        tk.Entry(frame, textvariable=self.groups_var).pack(fill="x", padx=6, pady=4)

        tk.Label(frame, text="Sessions (comma-separated)").pack(anchor="w", padx=6, pady=(6, 0))
        tk.Entry(frame, textvariable=self.sessions_var).pack(fill="x", padx=6, pady=4)

        # Output
        tk.Label(frame, text="Output root folder").pack(anchor="w", padx=6, pady=(6, 0))
        tk.Entry(frame, textvariable=self.out_var).pack(fill="x", padx=6, pady=4)
        tk.Button(frame, text="Browse…", command=self._browse_out).pack(anchor="e", padx=6, pady=(0, 6))

        # Hint
        tk.Label(
            frame,
            text="Expected layout: base / <Group> / <Session> / *.set",
            anchor="w",
            justify="left",
            fg="gray",
        ).pack(fill="x", padx=6, pady=(4, 8))

        # Aesthetic: brain image if available, else draw logo (resizable)
        holder = tk.Frame(frame, height=260)
        holder.pack_propagate(False)
        holder.pack(padx=6, pady=(4, 10), fill="both", expand=True)
        self._set_brain_image(holder)
        # Also listen to root resizes to refresh scaling
        self.root.bind("<Configure>", lambda e: self._render_brain_scaled())

    def _build_col2(self, frame: tk.Frame) -> None:
        # Filter settings
        row1 = tk.Frame(frame)
        row1.pack(fill="x", padx=6, pady=(8, 0))
        tk.Label(row1, text="Band-pass (Hz):").pack(side="left")
        tk.Entry(row1, width=7, textvariable=self.l_freq_var).pack(side="left", padx=(6, 4))
        tk.Label(row1, text="to").pack(side="left")
        tk.Entry(row1, width=7, textvariable=self.h_freq_var).pack(side="left", padx=(4, 0))

        row2 = tk.Frame(frame)
        row2.pack(fill="x", padx=6, pady=(8, 0))
        tk.Label(row2, text="Notch freqs (Hz, comma)").pack(side="left")
        tk.Entry(row2, width=20, textvariable=self.notch_var).pack(side="left", padx=(6, 0))

        # Options row: ASR and ICLabel
        row3 = tk.Frame(frame)
        row3.pack(fill="x", padx=6, pady=(8, 0))
        tk.Checkbutton(row3, text="Enable ASR", variable=self.asr_var).pack(side="left")
        tk.Checkbutton(row3, text="Enable ICLabel", variable=self.iclabel_var).pack(side="left", padx=(12, 0))

        row4 = tk.Frame(frame)
        row4.pack(fill="x", padx=6, pady=(8, 0))
        tk.Label(row4, text="ICLabel brain threshold").pack(side="left")
        tk.Entry(row4, width=6, textvariable=self.iclabel_thr_var).pack(side="left", padx=(6, 0))

        tk.Button(frame, text="Run Preprocessing", command=self._run_preprocessing_clicked).pack(padx=6, pady=(10, 6), anchor="w")

        # Status
        tk.Label(frame, text="Status").pack(anchor="w", padx=6)
        self.pre_log = tk.Text(frame, height=8, wrap="word", state="disabled")
        self.pre_log.pack(fill="x", padx=6, pady=(0, 6), expand=False)

        # Preview area (resizable)
        tk.Label(frame, text="Sample Pre/Post EEG traces").pack(anchor="w", padx=6)
        self.preview_holder = tk.Frame(frame, height=260)
        self.preview_holder.pack(fill="both", padx=6, pady=(0, 8), expand=True)
        self.preview_label = tk.Label(self.preview_holder, bd=0, anchor="center")
        self.preview_label.pack(fill="both", expand=True)
        self.preview_holder.bind("<Configure>", self._on_preview_resize)

    def _build_col3(self, frame: tk.Frame) -> None:
        # Options row: GPU + Stats
        opt = tk.Frame(frame)
        opt.pack(fill="x", padx=6, pady=(10, 4))
        tk.Checkbutton(opt, text="Use GPU if available", variable=self.use_gpu_var).pack(side="left")

        row_stats = tk.Frame(frame)
        row_stats.pack(fill="x", padx=6, pady=(4, 0))
        tk.Label(row_stats, text="Permutations").pack(side="left")
        tk.Entry(row_stats, width=8, textvariable=self.n_perm_var).pack(side="left", padx=(6, 12))
        tk.Label(row_stats, text="Cluster threshold (t)").pack(side="left")
        tk.Entry(row_stats, width=8, textvariable=self.cluster_thr_var).pack(side="left", padx=(6, 0))

        tk.Button(frame, text="Run Source Localization + Connectivity", command=self._run_source_clicked).pack(fill="x", padx=6, pady=(10, 6))
        tk.Button(frame, text="Get CSV for network analysis", command=self._export_csv_clicked).pack(fill="x", padx=6, pady=(0, 6))
        tk.Button(frame, text="Run Cluster-based permutation", command=self._run_stats_clicked).pack(fill="x", padx=6, pady=(0, 6))

        tk.Label(frame, text="Status").pack(anchor="w", padx=6)
        self.stage3_log = tk.Text(frame, height=16, wrap="word", state="disabled")
        self.stage3_log.pack(fill="both", padx=6, pady=(0, 8), expand=True)

    def _build_run_all_panel(self) -> None:
        """Build the Run All panel at the bottom of the window."""
        panel = tk.LabelFrame(self.root, text="Run All Pipeline Steps", fg="blue")
        panel.pack(fill="x", padx=8, pady=(0, 8))

        # Step selection checkboxes
        steps_frame = tk.Frame(panel)
        steps_frame.pack(fill="x", padx=6, pady=(6, 4))

        tk.Label(steps_frame, text="Select steps to run:").pack(side="left", padx=(0, 10))
        tk.Checkbutton(steps_frame, text="1. Preprocessing", variable=self.run_preprocessing_var).pack(side="left", padx=4)
        tk.Checkbutton(steps_frame, text="2. Source + Connectivity", variable=self.run_source_var).pack(side="left", padx=4)
        tk.Checkbutton(steps_frame, text="3. CSV Export", variable=self.run_csv_var).pack(side="left", padx=4)
        tk.Checkbutton(steps_frame, text="4. Statistics", variable=self.run_stats_var).pack(side="left", padx=4)

        # Run All button (prominent)
        btn_frame = tk.Frame(panel)
        btn_frame.pack(fill="x", padx=6, pady=(4, 8))

        self.run_all_btn = tk.Button(
            btn_frame,
            text="▶ RUN ALL SELECTED STEPS",
            command=self._run_all_clicked,
            bg="#4CAF50",
            fg="white",
            font=("TkDefaultFont", 10, "bold"),
            padx=20,
            pady=6,
        )
        self.run_all_btn.pack(side="left", padx=(0, 12))

        # Progress label
        self.run_all_progress = tk.Label(btn_frame, text="", fg="gray")
        self.run_all_progress.pack(side="left", fill="x", expand=True)

    # ---------- Actions ----------
    def _browse_base(self) -> None:
        d = filedialog.askdirectory(title="Select base folder (groups)")
        if d:
            self.base_var.set(d)

    def _browse_out(self) -> None:
        d = filedialog.askdirectory(title="Select output root")
        if d:
            self.out_var.set(d)

    def _design(self) -> StudyDesign:
        groups = [g.strip() for g in self.groups_var.get().split(",") if g.strip()]
        sessions = [s.strip() for s in self.sessions_var.get().split(",") if s.strip()]
        return StudyDesign(groups=groups, sessions=sessions)

    def _config(self) -> PipelineConfig:
        cfg = PipelineConfig()
        # Read preprocessing settings
        try:
            cfg.preprocessing.l_freq = float(self.l_freq_var.get())
            cfg.preprocessing.h_freq = float(self.h_freq_var.get())
        except Exception:
            pass
        try:
            txt = self.notch_var.get().strip()
            if txt:
                cfg.preprocessing.notch_freqs = tuple(float(x) for x in txt.split(",") if x.strip())
        except Exception:
            pass
        # Feature flags and thresholds
        cfg.preprocessing.use_asr = bool(self.asr_var.get())
        cfg.preprocessing.use_iclabel = bool(self.iclabel_var.get())
        try:
            cfg.preprocessing.iclabel_brain_threshold = float(self.iclabel_thr_var.get())
        except Exception:
            pass
        # Connectivity/compute options
        try:
            # Use GPU if available when True; CPU-only when False
            cfg.connectivity.use_gpu = None if bool(self.use_gpu_var.get()) else False
        except Exception:
            pass
        # Stats options
        try:
            # Ensure per-instance stats config exists
            if not hasattr(cfg, "stats") or getattr(cfg, "stats") is None:
                from pli_pipeline import StatsConfig  # local import to avoid cycles
                cfg.stats = StatsConfig()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            txt = self.n_perm_var.get().strip()
            if txt:
                n_perm = int(float(txt))
                if n_perm > 0:
                    cfg.stats.n_permutations = n_perm  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            txt = self.cluster_thr_var.get().strip()
            if txt:
                cfg.stats.cluster_threshold = float(txt)  # type: ignore[attr-defined]
            else:
                cfg.stats.cluster_threshold = None  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            cfg.output_root = Path(self.out_var.get()).expanduser()
            cfg.output_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Output folder error", str(exc))
        return cfg

    def _log_pre(self, msg: str) -> None:
        self.pre_log.configure(state="normal")
        self.pre_log.insert(tk.END, f"{msg}\n")
        self.pre_log.see(tk.END)
        self.pre_log.configure(state="disabled")
        self.root.update_idletasks()

    def _log_stage3(self, msg: str) -> None:
        self.stage3_log.configure(state="normal")
        self.stage3_log.insert(tk.END, f"{msg}\n")
        self.stage3_log.see(tk.END)
        self.stage3_log.configure(state="disabled")
        self.root.update_idletasks()

    def _validate_design_and_base(self) -> Optional[tuple[Path, StudyDesign, PipelineConfig]]:
        base = self.base_var.get().strip()
        if not base:
            messagebox.showwarning("Missing path", "Select a base folder first.")
            return None
        design = self._design()
        if not design.groups or not design.sessions:
            messagebox.showwarning("Design error", "Provide at least one group and one session.")
            return None
        cfg = self._config()
        return Path(base), design, cfg

    def _run_preprocessing_clicked(self) -> None:
        ok = self._validate_design_and_base()
        if not ok:
            return
        base, design, cfg = ok
        self._log_pre("Starting preprocessing …")
        summaries = run_preprocessing_for_design(base, design, cfg, progress_cb=self._log_pre)
        n = sum(1 for s in summaries if s.get("preprocessed_fif"))
        self._log_pre(f"Preprocessing complete. {n} files ready for source localization.")
        # Update preview if any PSD exists
        self._update_preview_image(cfg.output_root, design)

    def _run_source_clicked(self) -> None:
        ok = self._validate_design_and_base()
        if not ok:
            return
        base, design, cfg = ok
        self._log_stage3("Running source localization + connectivity …")
        summaries = run_source_and_connectivity_for_design(base, design, cfg, progress_cb=self._log_stage3)
        m = sum(len(s.get("pli_outputs", {})) for s in summaries)
        self._log_stage3(f"Source/Connectivity complete. Saved {m} band CSVs.")

    def _export_csv_clicked(self) -> None:
        ok = self._validate_design_and_base()
        if not ok:
            return
        _base, design, cfg = ok
        self._log_stage3("Aggregating PLI into Excel …")
        table = aggregate_csv_table(design, cfg, progress_cb=self._log_stage3)
        self._log_stage3(f"Saved study-level table to {cfg.output_root / 'PLI_Table.xlsx'} ({len(table)} rows)")
        messagebox.showinfo("Network table", f"Saved Excel to {cfg.output_root / 'PLI_Table.xlsx'}")

    def _run_stats_clicked(self) -> None:
        ok = self._validate_design_and_base()
        if not ok:
            return
        _base, design, cfg = ok
        self._log_stage3("Running cluster-based permutation (t-tests per edge) …")
        run_stats_for_design(design, cfg, progress_cb=self._log_stage3)
        self._log_stage3(f"Stats saved under {cfg.output_root / 'stats'}")
        messagebox.showinfo("Stats", f"Saved stats under {cfg.output_root / 'stats'}")

    def _run_all_clicked(self) -> None:
        """Run all selected pipeline steps sequentially."""
        # Validate inputs first
        ok = self._validate_design_and_base()
        if not ok:
            return
        base, design, cfg = ok

        # Check at least one step is selected
        steps_selected = (
            self.run_preprocessing_var.get()
            or self.run_source_var.get()
            or self.run_csv_var.get()
            or self.run_stats_var.get()
        )
        if not steps_selected:
            messagebox.showwarning("No steps selected", "Please select at least one step to run.")
            return

        # Disable the Run All button during execution
        self.run_all_btn.configure(state="disabled")
        self.run_all_progress.configure(text="Starting pipeline...", fg="blue")
        self.root.update_idletasks()

        errors = []
        total_steps = sum([
            self.run_preprocessing_var.get(),
            self.run_source_var.get(),
            self.run_csv_var.get(),
            self.run_stats_var.get(),
        ])
        current_step = 0

        try:
            # Step 1: Preprocessing
            if self.run_preprocessing_var.get():
                current_step += 1
                self.run_all_progress.configure(text=f"Step {current_step}/{total_steps}: Preprocessing...")
                self.root.update_idletasks()
                try:
                    self._log_pre("=" * 50)
                    self._log_pre("[RUN ALL] Starting preprocessing …")
                    summaries = run_preprocessing_for_design(base, design, cfg, progress_cb=self._log_pre)
                    n = sum(1 for s in summaries if s.get("preprocessed_fif"))
                    self._log_pre(f"[RUN ALL] Preprocessing complete. {n} files ready.")
                    self._update_preview_image(cfg.output_root, design)
                except Exception as e:
                    errors.append(f"Preprocessing: {e}")
                    self._log_pre(f"[RUN ALL] Preprocessing error: {e}")

            # Step 2: Source Localization + Connectivity
            if self.run_source_var.get():
                current_step += 1
                self.run_all_progress.configure(text=f"Step {current_step}/{total_steps}: Source + Connectivity...")
                self.root.update_idletasks()
                try:
                    self._log_stage3("=" * 50)
                    self._log_stage3("[RUN ALL] Running source localization + connectivity …")
                    summaries = run_source_and_connectivity_for_design(base, design, cfg, progress_cb=self._log_stage3)
                    m = sum(len(s.get("pli_outputs", {})) for s in summaries)
                    self._log_stage3(f"[RUN ALL] Source/Connectivity complete. Saved {m} band CSVs.")
                except Exception as e:
                    errors.append(f"Source/Connectivity: {e}")
                    self._log_stage3(f"[RUN ALL] Source/Connectivity error: {e}")

            # Step 3: CSV Export
            if self.run_csv_var.get():
                current_step += 1
                self.run_all_progress.configure(text=f"Step {current_step}/{total_steps}: CSV Export...")
                self.root.update_idletasks()
                try:
                    self._log_stage3("[RUN ALL] Aggregating PLI into Excel …")
                    table = aggregate_csv_table(design, cfg, progress_cb=self._log_stage3)
                    self._log_stage3(f"[RUN ALL] Saved table to {cfg.output_root / 'PLI_Table.xlsx'} ({len(table)} rows)")
                except Exception as e:
                    errors.append(f"CSV Export: {e}")
                    self._log_stage3(f"[RUN ALL] CSV Export error: {e}")

            # Step 4: Statistics
            if self.run_stats_var.get():
                current_step += 1
                self.run_all_progress.configure(text=f"Step {current_step}/{total_steps}: Statistics...")
                self.root.update_idletasks()
                try:
                    self._log_stage3("[RUN ALL] Running cluster-based permutation …")
                    run_stats_for_design(design, cfg, progress_cb=self._log_stage3)
                    self._log_stage3(f"[RUN ALL] Stats saved under {cfg.output_root / 'stats'}")
                except Exception as e:
                    errors.append(f"Statistics: {e}")
                    self._log_stage3(f"[RUN ALL] Statistics error: {e}")

        finally:
            # Re-enable the button
            self.run_all_btn.configure(state="normal")

        # Show completion status
        if errors:
            self.run_all_progress.configure(
                text=f"Completed with {len(errors)} error(s). Check logs for details.",
                fg="red"
            )
            error_msg = "\n".join(errors)
            messagebox.showwarning("Pipeline completed with errors", f"Errors encountered:\n{error_msg}")
        else:
            self.run_all_progress.configure(text="All steps completed successfully!", fg="green")
            messagebox.showinfo(
                "Pipeline Complete",
                f"All selected steps completed successfully!\n\nOutput saved to:\n{cfg.output_root}"
            )

    def _update_preview_image(self, output_root: Path, design: StudyDesign) -> None:
        # find a traces image from first available subject
        psd_path = None
        for g in design.groups:
            for s in design.sessions:
                d = output_root / g / s
                if not d.exists():
                    continue
                for subj in sorted(d.glob("*")):
                    cand = subj / "traces_before_after.png"
                    if cand.exists():
                        psd_path = cand
                        break
                if psd_path:
                    break
            if psd_path:
                break
        if not psd_path:
            return
        self.preview_path = psd_path
        try:
            if Image and ImageTk:
                self.preview_img_orig = Image.open(psd_path)
                self._render_preview_scaled()
            else:
                self.preview_label.configure(text=str(psd_path))
        except Exception:
            self.preview_label.configure(text=str(psd_path))

    def _draw_brain_logo(self, canvas: "tk.Canvas") -> None:
        # Simple stylized brain using ovals and arcs
        w = int(canvas["width"]) if str(canvas["width"]).isdigit() else 320
        h = int(canvas["height"]) if str(canvas["height"]).isdigit() else 180
        cx, cy = w // 2, h // 2
        # Outer lobes
        canvas.create_oval(cx - 120, cy - 60, cx - 10, cy + 60, fill="#f3c1d7", outline="#c98bb1")
        canvas.create_oval(cx + 10, cy - 60, cx + 120, cy + 60, fill="#f3c1d7", outline="#c98bb1")
        # Central seam
        canvas.create_line(cx, cy - 55, cx, cy + 55, fill="#b36a94", width=2, smooth=True)
        # Folds (gyri) as arcs
        for i, yy in enumerate(range(-40, 50, 20)):
            canvas.create_arc(cx - 115, cy + yy - 20, cx + 115, cy + yy + 20, start=0, extent=180, style=tk.ARC, outline="#b36a94", width=2)
        # Subtle shadow
        canvas.create_oval(cx - 130, cy - 70, cx + 130, cy + 70, outline="#e7a9c8")

    def _set_brain_image(self, parent: "tk.Widget") -> None:
        # Try to show BrainImage.png; fallback to drawing
        self.brain_holder = parent
        paths = [Path(__file__).parent / "BrainImage.png", Path.cwd() / "BrainImage.png"]
        for p in paths:
            try:
                if Image and ImageTk and p.exists():
                    self.brain_img_orig = Image.open(p)
                    self.brain_img_label = tk.Label(parent, bd=0)
                    self.brain_img_label.pack(fill="both", expand=True)
                    parent.bind("<Configure>", self._on_brain_resize)
                    self._render_brain_scaled()
                    return
            except Exception:
                continue
        # Fallback: draw stylized brain on resizable canvas
        self.brain_canvas = tk.Canvas(parent, width=320, height=180, bg="white", highlightthickness=1, highlightbackground="#ddd")
        self.brain_canvas.pack(fill="both", expand=True)
        self._draw_brain_logo(self.brain_canvas)
        self.brain_canvas.bind("<Configure>", lambda e: (self.brain_canvas.delete("all"), self._draw_brain_logo(self.brain_canvas)))

    # ---------- Dynamic resizers ----------
    def _on_brain_resize(self, event) -> None:
        self._render_brain_scaled()

    def _render_brain_scaled(self) -> None:
        if not (self.brain_img_orig and Image and ImageTk and self.brain_img_label):
            return
        w = max(50, int(self.brain_holder.winfo_width()))
        h = max(50, int(self.brain_holder.winfo_height()))
        try:
            img = self.brain_img_orig.copy()
            img.thumbnail((w, h))
            self.brain_img = ImageTk.PhotoImage(img)
            self.brain_img_label.configure(image=self.brain_img)
        except Exception:
            pass

    def _on_preview_resize(self, event) -> None:
        self._render_preview_scaled()

    def _render_preview_scaled(self) -> None:
        if not (self.preview_img_orig and Image and ImageTk and self.preview_label):
            return
        w = max(100, int(self.preview_holder.winfo_width()))
        h = max(100, int(self.preview_holder.winfo_height()))
        try:
            img = self.preview_img_orig.copy()
            img.thumbnail((w, h))
            self.preview_img = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_img)
        except Exception:
            pass


def launch_app() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    app = PLIPipelineApp()
    app.root.mainloop()


if __name__ == "__main__":
    launch_app()

