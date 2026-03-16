dataset_path = "/home/carlo/Download/ACDC/database/training"
preprocessed_2d_path = "/home/carlo/Projects/UT/acdc/artifacts/preprocessed_2d"

target_spacing_2d = (1.25, 1.25, -1.0)
patch_size_2d = (192, 192)
include_background_slices_2d = True
min_label_pixels_2d = 1
seed_2d = 42

preprocess_qc_limit_2d = 20
preprocess_qc_output_json_2d = "/home/carlo/Projects/UT/acdc/artifacts/preprocess_qc_report_2d.json"
preprocess_qc_output_figures_2d = "/home/carlo/Projects/UT/acdc/artifacts/preprocess_visual_qc_2d"
