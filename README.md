# VENV and LIBRARIES
We will use uv, it will handle packages and virtual enviroment:

- A virtual enviroment is like a box were you install all your packager. This way is not installed in all your system. This is good because each project may need different packages versions or can have conficts with other packages installed in the system. This way you have an isolated and easy to replicate enviroment.

1. create a virtual enviroment if you don't have one: uv venv
2. every time you need to install packages or run code, you need to be inside your venv. Otherwise:
- you will install the packages in the whole syste
- the code will not run because it relies on the packages inside the venv that if it is not activated the code can't see.
To activate the venv do: source .venv/bin/activate

3. To install a package use uv add .....
4. Every time you install a package, a line with the package name and version will be added in the pyproject.toml
Why? This is good because it keeps track of packages and versions needed for the project. This way if i want to get the exact libraries and versions
of the person who wrote the code i can just use the following command, always inside the venv (so yu have to activate it):
uv sync
this will syncronize you packages with the one on the pyproject.toml
5. To run code do: python path/to/pythonFile.py



# PREPROCESSING

1. SEMI-ISOTROPIC RESAMPLING
ACDC is anisotropic, x and y can vary from from 1.3-1.7 but the slice thickness (z)
is 5-8mm, if we would to an isotropic resamplic this would lead to a lot of artifacts. For this reason we only resample the xy to be 1.25x1.25

2. INTESITY NORMALIZATION (z-score per volume) 
Because MRI don't have an absolute scale and depends on the settings of the machine
we need to do intensity normalization. 
Also give faster convergence for the model because ....

3. PADDING TO FIXED INPUT SIZE
After slice extraction and in-plane resampling, tensors are padded to the configured
`patch_size_2d` so the network sees a stable input size without applying foreground crop.

- 4. DATA AUGMENTATION

TODO:
- check 1.3-1.7mm
- do we need to keep Orientationd in build_preprocess_transform function
- complete ... in 2. INTENSITY NORMALIZATION
- choose pad dimension based on desired input unet (SpatialPadD build_process_transform)
- 4. AUGMENTATION motivation (should be realistic and happen in real life)
- visualization


-Why 1.25x1.25?
-
