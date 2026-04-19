import os

from dotenv import load_dotenv


load_dotenv()

dataset_path = os.getenv("DATASET_PATH")
preprocessed_data_path = os.getenv(
    "PREPROCESSED_DATA_PATH",
    os.getenv("PREPROCESSED_2D_PATH", "artifacts/preprocessed_data"),
)
model_output_dir = os.getenv("MODEL_OUTPUT_DIR", "artifacts/models")
test_path = os.getenv("TEST_PATH")

# Slice preprocessing mode used across preprocess, train, and test.
# Options: "2d" | "2.5d"
preprocessing_mode = "2.5d"

target_spacing_2d = (1.25, 1.25, -1.0)
patch_size_2d = (320, 320)
include_background_slices_2d = True
min_label_pixels_2d = 1
seed_2d = 42

preprocess_qc_limit_2d = 20
preprocess_qc_output_json_2d = os.getenv(
    "PREPROCESS_QC_OUTPUT_JSON_2D",
    "artifacts/preprocess_qc_report_2d.json",
)
preprocess_qc_output_figures_2d = os.getenv(
    "PREPROCESS_QC_OUTPUT_FIGURES_2D",
    "artifacts/preprocess_visual_qc_2d",
)

preprocess_qc_output_json_2d5 = os.getenv(
    "PREPROCESS_QC_OUTPUT_JSON_2D5",
    "artifacts/preprocess_qc_report_2d5.json",
)
preprocess_qc_output_figures_2d5 = os.getenv(
    "PREPROCESS_QC_OUTPUT_FIGURES_2D5",
    "artifacts/preprocess_visual_qc_2d5",
)
