dataset_path = "C:\\Users\\Nils\\Documents\\Studie Nils\\Deep Learning 3D Images\\ACDC_data\\database\\training"
# Path for desktop
preprocessed_2d_path = "C:\\Users\\Nils\\Documents\\Studie Nils\\Deep Learning 3D Images\\ACDC_data\\preprocessed_2d"
# Path for laptop
# preprocessed_2d_path = "C:\\Users\\nilss\\Desktop\\ACDC_Project\\preprocessed_2d"

target_spacing_2d = (1.25, 1.25, -1.0)
patch_size_2d = (320, 320)
include_background_slices_2d = True
min_label_pixels_2d = 1
seed_2d = 42

preprocess_qc_limit_2d = 20
preprocess_qc_output_json_2d = "C:\\Users\\Nils\\Documents\\Studie Nils\\Deep Learning 3D Images\\ACDC_data\\artifacts\\preprocess_qc_report_2d.json"
preprocess_qc_output_figures_2d = "C:\\Users\\Nils\\Documents\\Studie Nils\\Deep Learning 3D Images\\ACDC_data\\artifacts\\preprocess_visual_qc_2d"
