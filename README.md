# accuracy-paradox-demo

Reproducible companion code for the Medium article **"Your Model Has 98% Accuracy and Is Completely Useless"**, a case study in class imbalance, why accuracy fails on skewed datasets, and how precision, recall, F1, and the precision-recall curve reveal what accuracy hides.

Read the article: `article.md` (or the published Medium link, added here once live).

## What's in here

A synthetic 98%/2% imbalanced classification dataset (styled after a medical-imaging screening problem), a baseline Random Forest, and three standard imbalance fixes: class weighting, SMOTE, and threshold tuning, evaluated honestly, including where a "standard fix" underperformed the baseline.

```
.
├── article.md                     # Full Medium article (markdown)
├── SETUP_GUIDE.md                 # Step-by-step: run this on your PC
├── GITHUB_GUIDE.md                # Repo naming and git push instructions
├── requirements.txt
├── src/
│   ├── imbalance_analysis.py      # End-to-end script: data, models, all figures
│   ├── make_banner.py             # Generates the article banner image
│   ├── make_equations.py          # Generates the equation images for Medium
│   └── make_table.py              # Generates the results table image for Medium
├── notebook/
│   └── imbalance_analysis.ipynb   # Same analysis, notebook form
└── figures/                       # All generated figures (PNG)
    ├── banner.png
    ├── fig1_accuracy_vs_recall.png
    ├── fig2_confusion_baseline.png
    ├── fig3_pr_curve_baseline.png
    ├── fig4_pr_curve_comparison.png
    ├── fig5_confusion_weighted.png
    ├── fig6_full_comparison.png
    ├── table_results_summary.png
    └── equations/
        ├── eq_accuracy.png
        ├── eq_recall.png
        ├── eq_precision.png
        └── eq_f1.png
```

## Results summary

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Predict majority always | 0.9748 | 0.0000 | 0.0000 | 0.0000 |
| Baseline (unweighted RF) | 0.9804 | 0.9667 | 0.2302 | 0.3718 |
| Threshold tuned | 0.9842 | 0.7474 | 0.5635 | **0.6425** |
| Class weighted | 0.9776 | 1.0000 | 0.1111 | 0.2000 |
| SMOTE | 0.9812 | 0.7286 | 0.4048 | 0.5204 |

Accuracy stays within a 1-point band across every method (97.5% to 98.4%), regardless of how well the model actually finds the minority class. That's the whole point. Precision, recall, and F1 are what actually separate a useful model from a useless one here.

## Run it locally

Full walkthrough with troubleshooting is in `SETUP_GUIDE.md`. Short version:

```bash
git clone https://github.com/<your-username>/accuracy-paradox-demo.git
cd accuracy-paradox-demo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd src
python imbalance_analysis.py    # regenerates every figure and prints the metrics table
python make_banner.py           # regenerates the banner image
python make_equations.py        # regenerates the Medium equation images
python make_table.py            # regenerates the Medium results table image
```

Everything is seeded (`random_state=42`) for exact reproducibility. Swap in your own dataset by replacing the `make_classification` call in `imbalance_analysis.py` with your own `X, y`. The rest of the pipeline (metrics, curves, fixes, figures) works unchanged on any binary classification problem.

## Push this to your own GitHub

See `GITHUB_GUIDE.md` for the suggested repository name, exact `git` commands, and an alternative using the GitHub CLI.

## Publishing to Medium

Medium doesn't render Markdown or LaTeX directly. The "Publishing notes for Medium" section at the bottom of `article.md` explains exactly how to insert the pre-rendered equation images and the results table image using Medium's native editor.

## Why this matters for medical imaging specifically

This is a direct extension of imbalance issues encountered in real diagnostic imaging work: CT-based classification where the disease-positive class is a small minority of scans, and where a model that quietly ignores the minority class can still report a deceptively strong headline accuracy. The checklist in the article, and the confusion-matrix-first habit behind it, is the same discipline that catches subject-level data leakage and validation-pipeline leaks before they inflate a reported result.

## License

Code: MIT. Article text: feel free to reference with attribution.
