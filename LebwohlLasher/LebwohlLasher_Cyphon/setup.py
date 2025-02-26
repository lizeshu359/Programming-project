from Cython.Build import cythonize
from setuptools import setup
import numpy as np

setup(
    name="lebwohl_lasher",
    ext_modules=cythonize("LebwohlLasher_cy.pyx", language_level="3"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)