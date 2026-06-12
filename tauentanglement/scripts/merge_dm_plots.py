"""
Merge gen, recoNu, recoRun3 plots from a model output directory into one PDF.
Each page groups all three versions of a given DM combination side by side.
Usage: python merge_dm_plots.py <model_dir> [output.pdf]
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


dm_order = [
    "DM0DM0", "DM0DM1", "DM0DM2", "DM0DM10",
    "DM1DM1", "DM1DM2", "DM1DM10",
    "DM2DM2", "DM2DM10", "DM10DM10",
    'DM100DM0', 'DM100DM1', 'DM100DM2', 'DM100DM10',
]

sections = [
    ("gen",      "Generator Level",         "#2166ac"),
    ("recoNu",   "Reconstructed Neutrino", "#1a9641"),
    ("recoRun3", "Previous Method",            "#d7191c"),
]


def format_dm_title(dm):
    idx = dm.index("DM", 2)
    return f"{dm[:idx]} - {dm[idx:]}"


def pdf_to_images(pdf_path, out_dir, prefix, dpi=150):
    pattern = os.path.join(out_dir, f"{prefix}_%03d.png")
    subprocess.run(
        ["gs", "-dBATCH", "-dNOPAUSE", "-q",
         "-sDEVICE=png16m", f"-r{dpi}",
         f"-sOutputFile={pattern}", pdf_path],
        check=True, capture_output=True,
    )
    return sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.png")))


def find_pdf(directory, dm):
    matches = glob.glob(os.path.join(directory, f"{dm}_*.pdf"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Combine DM plots side by side for easy comparison.")
    parser.add_argument("model_dir", help="Directory containing gen/, recoNu/, recoRun3/ subdirectories")
    parser.add_argument("--output", nargs="?", default=None, help="Output PDF (default: <model_dir>/combined_plots.pdf)")
    parser.add_argument("--compare", default=None, help="Compare recoNu from model_dir against recoNu in this directory")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    output_path = args.output or os.path.join(model_dir, "combined_plots.pdf")

    with tempfile.TemporaryDirectory() as tmpdir:
        with PdfPages(output_path) as pdf:
            for dm in dm_order:
                section_data = []

                if args.compare:
                    compare_dir = os.path.abspath(args.compare)
                    pairs = [
                        (os.path.join(model_dir, "recoNu"),  os.path.basename(model_dir),  "#2166ac", "New"),
                        (os.path.join(compare_dir, "recoNu"), os.path.basename(compare_dir), "#1a9641", "Compare"),
                        (os.path.join(model_dir, "recoRun3"),  "Run 3 Method",  "#d7191c", "Run 3 Method"),
                    ]
                    for section_dir, label, color, prefix_tag in pairs:
                        if not os.path.isdir(section_dir):
                            print(f"  Warning: recoNu not found in {section_dir}, skipping.", file=sys.stderr)
                            continue
                        pdf_path = find_pdf(section_dir, dm)
                        if pdf_path is None:
                            print(f"  Warning: no PDF for {dm} in {section_dir}, skipping.", file=sys.stderr)
                            continue
                        imgs = pdf_to_images(pdf_path, tmpdir, f"{dm}_recoNu_{prefix_tag}", dpi=150)
                        section_data.append((label, color, imgs))
                else:
                    for subdir, label, color in sections:
                        section_dir = os.path.join(model_dir, subdir)
                        if not os.path.isdir(section_dir):
                            continue
                        pdf_path = find_pdf(section_dir, dm)
                        if pdf_path is None:
                            print(f"  Warning: no PDF for {dm} in {subdir}, skipping section.", file=sys.stderr)
                            continue
                        imgs = pdf_to_images(pdf_path, tmpdir, f"{dm}_{subdir}", dpi=150)
                        section_data.append((label, color, imgs))

                if not section_data:
                    continue

                n_cols = len(section_data)
                n_pages = max(len(s[2]) for s in section_data)
                dm_title = format_dm_title(dm)
                print(f"  {dm_title}: {n_pages} page(s), {n_cols} version(s)")

                for page_idx in range(n_pages):
                    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 8, 6))
                    if n_cols == 1:
                        axes = [axes]

                    fig.suptitle(dm_title, fontsize=20, fontweight="bold", y=1.02)

                    for ax, (label, color, imgs) in zip(axes, section_data):
                        if page_idx < len(imgs):
                            img = mpimg.imread(imgs[page_idx])
                            ax.imshow(img)
                        ax.axis("off")
                        ax.set_title(label, fontsize=13, fontweight="bold",
                                     color="white", pad=4,
                                     bbox=dict(facecolor=color, edgecolor="none",
                                               boxstyle="round,pad=0.3"))

                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
