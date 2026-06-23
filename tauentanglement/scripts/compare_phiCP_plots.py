"""
Merge plots from up to three model directories into one PDF, grouping each
DM combination side by side.
Usage: python compare_phiCP_plots.py 'path1:"Name 1"' ['path2:"Name 2"'] ['path3:"Name 3"']
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

colors = ["#2166ac", "#1a9641", "#d7191c", "#fdae61", "#762a83"]


def parse_model_arg(arg):
    path, _, name = arg.partition(":")
    name = name.strip().strip('"').strip("'")
    if not name:
        name = os.path.basename(os.path.normpath(path))
    return os.path.abspath(path), name


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
    parser.add_argument("--model1", default=None, help='First model, as path:"Name", e.g. outputs_dir/phiCP/recoNu:"Model name"')
    parser.add_argument("--model2", default=None, help='Second model, as path:"Name" (optional)')
    parser.add_argument("--model3", nargs="?", default=None, help='Third model, as path:"Name" (optional)')
    parser.add_argument("--recoRun3", dest="reco_run3", default=None, help="Optional recoRun3 input path")
    parser.add_argument("--gen", default=None, help="Optional gen input path")
    parser.add_argument("--output", default=None, help="Output PDF (default: ./combined_plots.pdf)")
    args = parser.parse_args()

    models = [parse_model_arg(a) for a in (args.model1, args.model2, args.model3) if a is not None]
    if args.reco_run3 is not None:
        models.append((os.path.abspath(args.reco_run3), "Previous Method"))
    if args.gen is not None:
        models.append((os.path.abspath(args.gen), "Generator"))
    output_path = args.output or "combined_plots.pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        with PdfPages(output_path) as pdf:
            for dm in dm_order:
                section_data = []

                for (path, name), color in zip(models, colors):
                    if not os.path.isdir(path):
                        print(f"  Warning: directory not found: {path}, skipping.", file=sys.stderr)
                        continue
                    pdf_path = find_pdf(path, dm)
                    if pdf_path is None:
                        print(f"  Warning: no PDF for {dm} in {path}, skipping.", file=sys.stderr)
                        continue
                    safe_name = "".join(c if c.isalnum() else "_" for c in name)
                    imgs = pdf_to_images(pdf_path, tmpdir, f"{dm}_{safe_name}", dpi=150)
                    section_data.append((name, color, imgs))

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
