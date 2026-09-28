#!/usr/bin/env python3
"""Stop 'not random' from meaning 'leave-one-study-out'.

WHY THIS IS NEEDED NOW
----------------------
Five places select the cohort-shift scheme by excluding the random split:

    m[m["scheme"] != "random"]

That was correct while cv_metrics held exactly two schemes. Since
subject-grouped cross-validation was added, those selections silently pool
leave-one-study-out folds together with subject-grouped folds, which are an
easier evaluation. Anything computed from them is then a blend of two schemes
reported under one name.

Two of the five are not figures:

    s20_robustness.py          the robustness summary, and the per-fold
                               correlation with held-out outcome variance
    habitat/fold_heterogeneity.py   the gut side of the cross-habitat fold
                               analysis, which feeds the rho = 0.495 result

Their current outputs are clean, because they last ran before the third scheme
existed. They would be wrong the next time they run. This patch must therefore
be applied BEFORE regenerating any tables or figures.

The other three are panels D, E and F of the model figure.

WHAT IT DOES
------------
Adds a helper that names the cohort-shift scheme explicitly, preferring
leave_one_study_out, falling back to leave_one_country_out, and never
selecting random or subject_grouped. Behaviour with a two-scheme table is
unchanged, so figures built before the third scheme existed are reproduced.

    python patch_scheme_selection.py --src /mnt/d/w1_resistome_pipeline
    python patch_scheme_selection.py --src ... --revert
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HELPER = '''

# --- cohort-shift scheme selection -----------------------------------------
# Selecting by "not random" pooled leave-one-study-out with subject-grouped
# folds once a third scheme existed, reporting a blend of two evaluations
# under one name. The scheme is now named explicitly.
COHORT_SHIFT_SCHEMES = ("leave_one_study_out", "leave_one_country_out")


def cohort_shift_scheme(schemes) -> str:
    """The scheme that withholds whole cohorts, by name, never by exclusion."""
    present = list(dict.fromkeys(schemes))
    for s in COHORT_SHIFT_SCHEMES:
        if s in present:
            return s
    other = [s for s in present if s not in ("random", "subject_grouped")]
    return other[0] if other else (present[0] if present else "leave_one_study_out")


def only_cohort_shift(df):
    """Rows for the cohort-shift scheme alone."""
    if "scheme" not in df.columns:
        return df
    return df[df["scheme"] == cohort_shift_scheme(df["scheme"].unique())]
'''

EDITS = {
    "src/s07_figures.py": [
        ('            other = [s_ for s_ in schemes if s_ != "random"][0]',
         '            other = cohort_shift_scheme(schemes)'),
        ('        loso = m_rho[m_rho["scheme"] != "random"]',
         '        loso = only_cohort_shift(m_rho)'),
        ('            q = q[q["scheme"] != "random"] if (q["scheme"] != "random").any() else q',
         '            q = only_cohort_shift(q) if len(only_cohort_shift(q)) else q'),
    ],
    "src/s20_robustness.py": [
        ('    sub = m[(m["scheme"] != "random") & (m["model"] != "mean")]',
         '    sub = only_cohort_shift(m)\n    sub = sub[sub["model"] != "mean"]'),
    ],
    "habitat/fold_heterogeneity.py": [
        ('    m = m[(m["scheme"] != "random") & (m["model"] != "mean")]',
         '    m = only_cohort_shift(m)\n    m = m[m["model"] != "mean"]'),
    ],
}

SUFFIX = ".pre_scheme_patch"


def insert_helper(text: str) -> str:
    """Put the helper after the imports, before the first def/class."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(("def ", "class ")) and i > 0:
            return "".join(lines[:i]) + HELPER + "\n\n" + "".join(lines[i:])
    return text + HELPER


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="pipeline root (the directory holding src/ and habitat/)")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        for rel in EDITS:
            bak = args.src / (rel + SUFFIX)
            if bak.exists():
                shutil.copy2(bak, args.src / rel)
                print(f"reverted {rel}")
            else:
                print(f"no backup for {rel}")
        return 0

    for rel, edits in EDITS.items():
        p = args.src / rel
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 2
        s = p.read_text()
        if "def cohort_shift_scheme" in s:
            print(f"skip {rel}: already patched")
            continue
        for old, _ in edits:
            if s.count(old) != 1:
                print(f"ERROR: in {rel}, anchor appears {s.count(old)} times:\n"
                      f"  {old.strip()[:70]}", file=sys.stderr)
                return 2
        bak = args.src / (rel + SUFFIX)
        if not bak.exists():
            shutil.copy2(p, bak)
        for old, new in edits:
            s = s.replace(old, new)
        p.write_text(insert_helper(s))
        print(f"patched {rel} ({len(edits)} site(s))")

    print("\nRegenerate in this order once patched:")
    print("  python src/s20_robustness.py --pangenome <evidence.tsv> --habitats <dir>")
    print("  python habitat/fold_heterogeneity.py ...   # as in your original command")
    print("  python src/s07_figures.py")
    print("  python make_manuscript_figures.py --supplementary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
