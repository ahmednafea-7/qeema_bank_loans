# ============================================================
# Run this as a NEW CELL at the end of your Colab notebook,
# AFTER cell 16 (tuning) and cell 14 (regression) have run.
# It saves the two fitted pipelines you already trained.
# ============================================================
import joblib
import sklearn

joblib.dump(best_tuned_model, "classification_pipeline.joblib")   # from cell 16
joblib.dump(rf_reg_pipe, "regression_pipeline.joblib")             # from cell 14

print("Saved 2 files.")
print("scikit-learn version used to train:", sklearn.__version__)
print("^ Match this exact version in requirements.txt or the pickles may fail to load.")

# Then, in the Colab left sidebar -> Files, download both .joblib files
# (or use the snippet below to download them directly):
from google.colab import files
files.download("classification_pipeline.joblib")
files.download("regression_pipeline.joblib")
