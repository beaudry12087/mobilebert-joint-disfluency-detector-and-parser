from Cython.Build import cythonize
from setuptools import setup, Extension
import os

# Full path to the .pyx file
pyx_file = "/Users/beaudry/Documents/mobilebert-joint-disfluency-detector-and-parser/src/chart_helper.pyx"

# Ensure the file exists before trying to cythonize it
if not os.path.exists(pyx_file):
    raise ValueError(f"File {pyx_file} does not exist!")

# Cythonize the .pyx file
ext_modules = cythonize([pyx_file])

# Setup the extension
setup(
    name="chart_helper",
    ext_modules=ext_modules,
)
