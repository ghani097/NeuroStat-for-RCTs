"""
Generate publication-ready Word document for sections 1.4.6.1 and 1.4.6.2.

Renders PlantUML diagrams via the public PlantUML server and embeds them in a
DOCX file using python-docx.
"""

from __future__ import annotations

import io
import zlib
import base64
import string
import sys
import textwrap
from pathlib import Path

import requests
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ---------------------------------------------------------------------------
# PlantUML encoding helpers
# ---------------------------------------------------------------------------

# PlantUML uses standard base64 but with a different alphabet, and strips
# the 2-byte zlib header and 4-byte Adler32 checksum from the compressed data.
_B64_ALPHABET    = (string.ascii_uppercase + string.ascii_lowercase
                    + string.digits + "+/")
_PUML_ALPHABET   = (string.digits + string.ascii_uppercase
                    + string.ascii_lowercase + "-_")
_B64_TO_PUML = bytes.maketrans(
    _B64_ALPHABET.encode("ascii"), _PUML_ALPHABET.encode("ascii")
)


def plantuml_encode(text: str) -> str:
    """Encode PlantUML source for use in the public server URL."""
    zlibbed = zlib.compress(text.encode("utf-8"), 9)
    # Strip 2-byte zlib header and 4-byte Adler32 trailer → raw DEFLATE stream
    deflated = zlibbed[2:-4]
    b64 = base64.b64encode(deflated)
    return b64.translate(_B64_TO_PUML).decode("ascii")


def render_plantuml(source: str, output_path: Path, fmt: str = "png") -> Path:
    """
    Render a PlantUML diagram via the public server and save to output_path.
    Returns the saved path.
    """
    encoded = plantuml_encode(source)
    url = f"https://www.plantuml.com/plantuml/{fmt}/{encoded}"
    print(f"  Fetching: {url[:80]}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    print(f"  Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# PlantUML diagram sources
# ---------------------------------------------------------------------------

DIAGRAM_1461 = """
@startuml
skinparam backgroundColor #FFFFFF
skinparam ArrowColor #2C5F8A
skinparam ActivityBorderColor #2C5F8A
skinparam ActivityBackgroundColor #EAF3FB
skinparam ActivityFontSize 12
skinparam ActivityFontName Arial
skinparam shadowing false
skinparam RoundCorner 10

title <b>Section 1.4.6.1 - EEG Pre-processing Pipeline</b>\\n(Input -> Pre-processing -> Output)

|#EAF3FB| <b>INPUT</b> |
start
:Raw EEG Recordings\\n.set / .fdt files (EEGLAB format)\\nMultiple subjects, sessions, groups;

|#EAF3FB| <b>PRE-PROCESSING</b> |
:Step 1 - Electrode Montage Assignment\\nHierarchical matching: BioSemi64/32/128/256,\\nstandard_1005, 1020, EasyCap-M1, GSN-HydroCel\\nAcceptance threshold: >= 80% channel coverage;

:Step 2 - Bad / Dead Channel Detection\\nVariance z-score criterion (threshold = 4.0 SD)\\nChannels exceeding threshold marked as bad;

:Step 3 - Bad Channel Reconstruction\\nSpherical Spline Interpolation\\n(Perrin et al., 1989);

:Step 4 - Artifact Removal\\nTwo mutually exclusive pathways:\\n[A] GEDAI (see Section 1.4.6.2)\\n[B] ASR + ICA  (Traditional pathway);

:Step 5 - Temporal Filtering\\nFIR Bandpass 1-40 Hz (Hamming window)\\nNotch filter at mains frequency (50 / 60 Hz + harmonics);

:Step 6 - Re-referencing\\nCommon Average Reference;

:Step 7 - Epoching\\nSegmentation into 0.5-second fixed-length epochs\\nConcatenated across trials;

|#EAF3FB| <b>OUTPUT</b> |
:Clean EEG Epochs\\n0.5-second epochs | concatenated .mat files\\nDiagnostic visualisations + preprocessing log\\nBand-specific PLI connectivity matrices;
stop

@enduml
"""

DIAGRAM_1462 = """
@startuml
skinparam backgroundColor #FFFFFF
skinparam ArrowColor #1A5276
skinparam ActivityBorderColor #1A5276
skinparam ActivityBackgroundColor #EBF5FB
skinparam ActivityFontSize 12
skinparam ActivityFontName Arial
skinparam shadowing false
skinparam RoundCorner 10

title <b>Section 1.4.6.2 - GEDAI Pre-processing Pipeline</b>\\n(Input -> GEDAI Denoising -> Output)

|#EBF5FB| <b>INPUT</b> |
start
:Raw Broadband EEG\\nMNE Raw object | all EEG channels\\nNo prior temporal filtering applied;

|#EBF5FB| <b>GEDAI DENOISING</b> |
:Compute Leadfield Matrix\\nPhysics-based forward model\\n(fsaverage BEM, standard_1005 montage)\\nEncodes expected spatial patterns of neural sources;

:Compute Reference Covariance Matrix\\nDerived from the leadfield matrix\\nFallback: identity-based -> data-driven;

:Compute Data Covariance Matrix\\nSliding windows: 2.0 s | 50% overlap;

:Solve Generalised Eigenvalue Problem\\nW = argmax [ W^T * C_data * W ]\\nsubject to: W^T * C_ref * W = I\\nSeparates brain signal from non-brain artefacts;

:SENSAI Threshold Selection\\nGrid search using Signal & Noise\\nSubspace Alignment Index (SENSAI)\\nNoise multiplier = 3.0  (balanced setting);

:Project Out Artefact Subspace\\nArtefact components identified and removed\\nClean signal reconstructed in sensor space;

|#EBF5FB| <b>OUTPUT</b> |
:Denoised Broadband EEG\\nArtefact subspace removed | brain signal preserved\\nDiagnostic metadata: variance removed, SNR improvement\\nReady for: filtering -> re-reference -> epoching -> PLI;
stop

@enduml
"""

# ---------------------------------------------------------------------------
# Word document helpers
# ---------------------------------------------------------------------------

def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def add_body(doc: Document, text: str) -> None:
    """Add justified body paragraph."""
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in p.runs:
        run.font.size = Pt(11)
        run.font.name = "Calibri"


def add_figure(doc: Document, img_path: Path, caption: str, width_in: float = 5.5) -> None:
    """Add an image with a centred caption."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_img.add_run()
    run.add_picture(str(img_path), width=Inches(width_in))

    p_cap = doc.add_paragraph(caption)
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p_cap.runs:
        run.font.size = Pt(9)
        run.font.italic = True
        run.font.name = "Calibri"


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"

    # Header row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for run in hdr_cells[i].paragraphs[0].runs:
            run.font.bold = True
            run.font.size = Pt(10)

    # Data rows
    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = val
            for run in cells[c_idx].paragraphs[0].runs:
                run.font.size = Pt(10)


# ---------------------------------------------------------------------------
# Section text
# ---------------------------------------------------------------------------

SEC_1461_INTRO = (
    "Electroencephalography (EEG) recording is inherently susceptible to a range of "
    "biological and environmental artefacts, including ocular transients, myogenic "
    "activity, cardiac interference, and mains-frequency line noise. Pre-processing "
    "constitutes a critical preliminary stage in any EEG analysis pipeline, serving "
    "to attenuate these contaminating signals while preserving the neurophysiological "
    "information of interest. The following subsection describes the software "
    "environment, bad-channel reconstruction algorithm, the overall processing "
    "sequence, and the expected output format employed in the present study."
)

SEC_1461_SOFTWARE = (
    "Raw EEG data were acquired and stored in EEGLAB format (Delorme & Makeig, 2004; "
    "Swartz Center for Computational Neuroscience, UC San Diego), a widely adopted "
    "MATLAB-based toolbox providing standardised file structures (.set header / .fdt "
    "binary data). It should be noted that different software implementations of "
    "nominally identical preprocessing operations (e.g., band-pass filtering, ICA) "
    "can yield quantitatively different outputs owing to differences in numerical "
    "precision, default parameter selections, and internal algorithmic choices "
    "(Bigdely-Shamlo et al., 2015). To ensure transparency and reproducibility, all "
    "downstream analyses were executed in Python using MNE-Python (version ≥ 1.6; "
    "Gramfort et al., 2013), an open-source library providing documented, "
    "version-controlled implementations of all pre-processing steps. Reading of "
    "EEGLAB .set files was performed natively via MNE-Python's `read_raw_eeglab` "
    "function, preserving channel metadata and event structure without conversion "
    "loss. The two-software architecture — EEGLAB for data capture, MNE-Python for "
    "offline analysis — is consistent with recommended best practices for clinical "
    "EEG research (Luck, 2014; Clayson et al., 2021)."
)

SEC_1461_BAD_CHANNEL = (
    "Channels exhibiting abnormal amplitude characteristics — indicative of electrode "
    "bridging, gel desiccation, amplifier saturation, or physical damage to the "
    "electrode lead — were identified prior to artefact removal using a "
    "variance-based z-score criterion. Per-channel variance was computed across the "
    "full continuous recording, and any channel whose variance deviated by more than "
    "4.0 standard deviations from the cross-channel mean was flagged as bad. This "
    "threshold is consistent with standard automated bad-channel detection "
    "approaches (Nolan et al., 2010; Bigdely-Shamlo et al., 2015).\n\n"
    "Flagged channels were subsequently reconstructed using spherical spline "
    "interpolation (Perrin et al., 1989). Spherical spline interpolation models the "
    "electrical potential on the scalp surface as a linear combination of basis "
    "functions defined on the unit sphere, providing a physically principled estimate "
    "of the potential at missing electrode locations from the surrounding recorded "
    "channels. This approach has been shown to produce more accurate reconstructions "
    "than simple distance-weighted averaging, particularly at scalp locations with "
    "sparse neighbouring electrodes (Perrin et al., 1989)."
)

SEC_1461_PROCESS = (
    "The pre-processing pipeline accepted raw EEGLAB .set files as input and applied "
    "the following sequential operations: (1) electrode montage assignment via a "
    "hierarchical matching algorithm that tested candidate standard montages "
    "(BioSemi 64/32/128/256, extended 10-05, standard 10-20, EasyCap M1, and EGI "
    "HydroCel) in order of decreasing channel count, accepting the first montage "
    "achieving ≥ 80% three-dimensional position coverage; (2) automated bad-channel "
    "detection and spherical spline interpolation as described above; "
    "(3) artefact suppression via either the GEDAI pathway or the Traditional "
    "ASR + ICA pathway (see Section 1.4.6.2); (4) finite impulse response (FIR) "
    "band-pass filtering (1–40 Hz, Hamming window) with optional notch filtering at "
    "the local mains frequency (50 or 60 Hz) and its harmonics; "
    "(5) re-referencing to the common average; and (6) segmentation into fixed-length "
    "epochs. Figure 1 illustrates the complete high-level data flow."
)

SEC_1461_OUTPUT = (
    "The output of the pre-processing stage consisted of clean EEG epochs of "
    "0.5-second duration, concatenated across trials and saved as MATLAB-compatible "
    ".mat files for downstream feature extraction and statistical analysis. Each "
    "output file was accompanied by a preprocessing log documenting the number of "
    "rejected channels, the artefact suppression pathway applied, the number of "
    "excluded ICA components (where applicable), and summary quality-control "
    "visualisations including power spectral density plots and topographic maps of "
    "channel-level signal quality. These outputs provided full provenance "
    "documentation in support of transparent reporting standards for clinical EEG "
    "research (Clayson et al., 2021)."
)

SEC_1462_INTRO = (
    "The Generalised Eigenvalue Decomposition-based Artefact Identification "
    "(GEDAI; Ros et al., 2025) method was selected as the primary automated "
    "EEG cleaning algorithm in the present study. GEDAI belongs to a family of "
    "source-separation approaches that exploit spatial filtering to attenuate "
    "non-neural signal components, but uniquely incorporates a physics-informed "
    "reference covariance matrix derived from the electromagnetic leadfield, enabling "
    "principled separation of brain activity from artefact without requiring "
    "manual component labelling or supervised training data."
)

SEC_1462_PROCESS = (
    "The GEDAI pipeline, implemented via the official `gedai` Python package "
    "(https://github.com/neurotuning/gedai), operated as follows. First, a "
    "physics-based reference covariance matrix was constructed from the electromagnetic "
    "leadfield computed on the fsaverage template brain (boundary element model; "
    "three-shell conductivities: scalp = 0.3, skull = 0.006, brain = 0.3 S/m). "
    "This leadfield encoded the expected spatial projection patterns of cortical "
    "dipolar sources onto the scalp electrodes, providing a data-independent prior "
    "for what 'brain-like' spatial covariance structure should look like. Second, a "
    "sliding-window data covariance matrix was computed across the raw (unfiltered) "
    "EEG in 2.0-second segments with 50% overlap. Third, a generalised eigenvalue "
    "problem was solved that identified a spatial filter subspace maximally "
    "aligned with the data covariance relative to the leadfield-derived reference "
    "covariance. Eigenvectors associated with eigenvalues substantially exceeding "
    "the reference — and therefore representing non-neural spatial patterns — "
    "were identified as artefact subspaces. Fourth, the optimal eigenvalue threshold "
    "was determined via grid search using the SENSAI (Signal and Noise Subspace "
    "Alignment Index) criterion with a noise multiplier of 3.0, representing a "
    "balanced setting between aggressive artefact suppression and signal preservation. "
    "Finally, the identified artefact components were projected out of the data, "
    "and the cleaned signal was reconstructed in sensor space. A fallback hierarchy "
    "was implemented: if the leadfield-based reference covariance failed (e.g., due "
    "to non-standard channel nomenclature), the algorithm sequentially attempted an "
    "identity-based and then a data-driven reference covariance. Critically, GEDAI "
    "was applied to the broadband signal prior to temporal filtering, preserving "
    "broad-band artefact signatures that would otherwise be truncated by band-pass "
    "filtering. Figure 2 illustrates the GEDAI processing sequence."
)

SEC_1462_WHY = (
    "Several fully automated EEG cleaning pipelines were considered for this study, "
    "including PREP (Bigdely-Shamlo et al., 2015), HAPPE (Gabard-Durnam et al., "
    "2018), FASTER (Nolan et al., 2010), ADJUST (Mognon et al., 2011), and "
    "Artefact Subspace Reconstruction (ASR; Mullen et al., 2015) combined with "
    "ICLabel-based ICA (Pion-Tonachini et al., 2019). GEDAI was selected for the "
    "following reasons:\n\n"
    "(i) Physics-informed reference. Unlike purely data-driven methods (HAPPE, FASTER, "
    "ADJUST) that classify artefacts on the basis of statistical heuristics trained "
    "on independent datasets, GEDAI's reference covariance is derived directly from "
    "the electromagnetic leadfield. This ensures that the artefact-versus-brain "
    "distinction reflects first-principles biophysics rather than assumptions about "
    "the statistical properties of 'typical' EEG, which may not generalise to "
    "clinical populations or novel recording environments.\n\n"
    "(ii) Fully unsupervised operation. GEDAI requires no manually labelled training "
    "data and applies no pre-trained classifier (cf. ICLabel, ADJUST). This removes "
    "a potential source of systematic bias when the demographics or recording "
    "conditions of the study sample differ from those used to train the classifier. "
    "The absence of manual component labelling also eliminates operator-dependent "
    "variability, which is particularly important in multi-site RCT contexts.\n\n"
    "(iii) Automatic threshold selection. The SENSAI criterion provides a principled, "
    "objective basis for selecting the artefact rejection threshold without requiring "
    "tuning per dataset or per subject. In contrast, ASR requires selection of a "
    "cutoff parameter (default: 20 standard deviations) that has been shown to affect "
    "results substantially across parameter values (Chang et al., 2020).\n\n"
    "(iv) Preservation of neural signal. By operating on the broadband signal and "
    "confining suppression to the artefact subspace identified relative to the "
    "leadfield reference, GEDAI avoids the distortion of neural frequency content "
    "that can accompany aggressive ASR or high-pass filtering strategies (Widmann "
    "et al., 2015).\n\n"
    "(v) Empirical performance. In the internal validation of the present pipeline "
    "(Section 2.5), GEDAI produced competitive or superior recovery of planted "
    "phase-coupling patterns relative to the Traditional ASR + ICA pathway under "
    "moderate and heavy artefact conditions, supporting its selection for primary "
    "analysis."
)

REFERENCES = (
    "Barry, R. J., Clarke, A. R., Johnstone, S. J., Magee, C. A., & Rushby, J. A. "
    "(2007). EEG differences between eyes-closed and eyes-open resting conditions. "
    "Clinical Neurophysiology, 118(12), 2765–2773.\n\n"
    "Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). "
    "The PREP pipeline: Standardized preprocessing for large-scale EEG analysis. "
    "Frontiers in Neuroinformatics, 9, 16.\n\n"
    "Chang, C.-Y., Hsu, S.-H., Pion-Tonachini, L., & Jung, T.-P. (2020). Evaluation "
    "of artifact subspace reconstruction for automatic artifact components removal "
    "in multi-channel EEG recordings. IEEE Transactions on Biomedical Engineering, "
    "67(4), 1114–1121.\n\n"
    "Clayson, P. E., Baldwin, S. A., Larson, M. J., et al. (2021). The data are out "
    "there, but are we asking the right questions? Measurement considerations for "
    "ERP research. Psychophysiology, 58(2), e13722.\n\n"
    "Delorme, A., & Makeig, S. (2004). EEGLAB: An open source toolbox for analysis "
    "of single-trial EEG dynamics including independent component analysis. "
    "Journal of Neuroscience Methods, 134(1), 9–21.\n\n"
    "Gabard-Durnam, L. J., Mendez Leal, A. S., Wilkinson, C. L., & Levin, A. R. "
    "(2018). The Harvard Automated Processing Pipeline for Electroencephalography "
    "(HAPPE): Standardized processing software for developmental and high-artifact "
    "data. Frontiers in Neuroscience, 12, 97.\n\n"
    "Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG data analysis "
    "with MNE-Python. Frontiers in Neuroscience, 7, 267.\n\n"
    "Luck, S. J. (2014). An Introduction to the Event-Related Potential Technique "
    "(2nd ed.). MIT Press.\n\n"
    "Mognon, A., Jovicich, J., Bruzzone, L., & Buiatti, M. (2011). ADJUST: An "
    "automatic EEG artifact detector based on the joint use of spatial and temporal "
    "features. Psychophysiology, 48(2), 229–240.\n\n"
    "Mullen, T. R., Kothe, C. A. E., Chi, Y. M., et al. (2015). Real-time "
    "neuroimaging and cognitive monitoring using wearable dry EEG. IEEE Transactions "
    "on Biomedical Engineering, 62(11), 2553–2567.\n\n"
    "Nolan, H., Whelan, R., & Reilly, R. B. (2010). FASTER: Fully automated "
    "statistical thresholding for EEG artifact rejection. Journal of Neuroscience "
    "Methods, 192(1), 152–162.\n\n"
    "Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F. (1989). Spherical "
    "splines for scalp potential and current density mapping. "
    "Electroencephalography and Clinical Neurophysiology, 72(2), 184–187.\n\n"
    "Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An "
    "automated electroencephalographic independent component classifier, dataset, "
    "and website. NeuroImage, 198, 181–197.\n\n"
    "Ros, T., Keshmiri, S., Baur, R., et al. (2025). Return of the GEDAI: "
    "Unsupervised EEG denoising based on leadfield filtering. bioRxiv "
    "10.1101/2025.10.04.680449.\n\n"
    "Widmann, A., Schröger, E., & Maess, B. (2015). Digital filter design for "
    "electrophysiological data — a practical approach. Journal of Neuroscience "
    "Methods, 250, 34–46."
)

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_document(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Use pre-rendered matplotlib diagrams (render_diagrams.py) ---
    diagram_dir = out_dir / "diagrams"
    img1 = diagram_dir / "fig1_eeg_preprocessing.png"
    img2 = diagram_dir / "fig2_gedai_pipeline.png"

    if not img1.exists() or not img2.exists():
        raise FileNotFoundError(
            "Run render_diagrams.py first to generate the diagram images."
        )
    print(f"Using diagrams from: {diagram_dir}")

    # --- Build document ---
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)

    # Default Normal style
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    # -------------------------------------------------------------------------
    # SECTION 1.4.6.1
    # -------------------------------------------------------------------------
    add_heading(doc, "1.4.6.1  EEG Cleaning Pre-processing", level=2)

    add_body(doc, SEC_1461_INTRO)

    add_heading(doc, "Software", level=3)
    add_body(doc, SEC_1461_SOFTWARE)

    add_heading(doc, "Reconstruction Algorithm for Bad / Dead Channels", level=3)
    add_body(doc, SEC_1461_BAD_CHANNEL)

    add_heading(doc, "High-Level Process (Input → Pre-processing → Output)", level=3)
    add_body(doc, SEC_1461_PROCESS)

    add_figure(
        doc, img1,
        "Figure 1. High-level EEG pre-processing pipeline. "
        "Raw EEGLAB .set files are passed through montage assignment, bad-channel "
        "detection and spherical spline reconstruction, artefact removal (GEDAI or "
        "Traditional pathway), band-pass and notch filtering, average re-referencing, "
        "and epoching to yield clean 0.5-second EEG epochs saved as .mat files.",
    )

    add_heading(doc, "Output", level=3)
    add_body(doc, SEC_1461_OUTPUT)

    # Comparison table
    doc.add_paragraph()
    add_heading(doc, "Table 1.  Key pre-processing parameters", level=4)
    add_table(
        doc,
        headers=["Parameter", "Value / Method", "Reference"],
        rows=[
            ["Bad-channel z-score threshold", "4.0 SD", "Bigdely-Shamlo et al., 2015"],
            ["Reconstruction method", "Spherical spline interpolation", "Perrin et al., 1989"],
            ["Band-pass filter", "1–40 Hz, FIR (Hamming)", "MNE-Python defaults"],
            ["Notch filter", "50 Hz (+ 100 Hz harmonic)", "Local mains frequency"],
            ["Reference", "Common average", "Standard practice"],
            ["Epoch length", "0.5 s", "Study protocol"],
            ["Output format", "Concatenated .mat file", "Study protocol"],
        ],
    )
    doc.add_paragraph()

    # -------------------------------------------------------------------------
    # SECTION 1.4.6.2
    # -------------------------------------------------------------------------
    add_heading(doc, "1.4.6.2  GEDAI – EEG Cleaning Pre-processing Pipeline", level=2)

    add_body(doc, SEC_1462_INTRO)

    add_heading(doc, "High-Level Process (Input → Pre-processing → Output)", level=3)
    add_body(doc, SEC_1462_PROCESS)

    add_figure(
        doc, img2,
        "Figure 2. GEDAI pre-processing pipeline. "
        "Broadband raw EEG is denoised by solving a generalised eigenvalue problem "
        "that contrasts the data covariance against a physics-informed leadfield "
        "reference covariance. The SENSAI criterion automatically selects the "
        "optimal artefact rejection threshold. The resulting clean signal is "
        "passed forward to temporal filtering, re-referencing, and epoching.",
    )

    add_heading(doc, "Why GEDAI? Rationale vs. Competing Automated Pipelines", level=3)
    add_body(doc, SEC_1462_WHY)

    # Comparison table
    doc.add_paragraph()
    add_heading(doc, "Table 2.  Comparison of automated EEG cleaning pipelines", level=4)
    add_table(
        doc,
        headers=["Pipeline", "Approach", "Supervised?", "Threshold selection", "Key limitation"],
        rows=[
            ["GEDAI", "Generalised EVD + leadfield reference", "No", "SENSAI (automatic)", "Requires leadfield / forward model"],
            ["ASR", "PCA-based burst removal", "No", "Manual cutoff (default 20 SD)", "Cutoff sensitivity; can distort ERP"],
            ["ASR + ICLabel", "ASR then ICA classification", "Yes (ICLabel CNN)", "ICLabel threshold (p > 0.85)", "CNN trained on specific ICA flavour"],
            ["HAPPE", "EEGLAB-based multi-step pipeline", "Yes (ADJUST/ICLabel)", "Semi-manual", "MATLAB dependency; slower"],
            ["FASTER", "Statistical thresholding (z-score)", "No", "z-score per metric", "No physics prior; heuristic metrics"],
            ["PREP", "Robust referencing + interpolation", "No", "Adaptive", "No ICA / subspace cleaning step"],
        ],
    )
    doc.add_paragraph()

    # -------------------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------------------
    add_heading(doc, "References", level=2)
    p = doc.add_paragraph(REFERENCES)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.size = Pt(10)
        run.font.name = "Calibri"

    # --- Save ---
    out_path = out_dir / "EEG_Preprocessing_Sections_1461_1462_FINAL.docx"
    doc.save(str(out_path))
    print(f"\nDocument saved: {out_path}")
    return out_path


if __name__ == "__main__":
    out = Path(__file__).parent
    build_document(out)
