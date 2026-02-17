# VENV and LIBRARIES aaaa
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

