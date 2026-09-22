from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = cythonize([
    Extension(
        "sdf._meshing",
        ["sdf/_meshing.pyx"],
        include_dirs=[numpy.get_include()]
    )
])

setup(
    name='sdf',
    version='0.1',
    description='Generate 3D meshes from signed distance functions.',
    author='Michael Fogleman',
    author_email='michael.fogleman@gmail.com',
    packages=['sdf'],
    install_requires=[
        'matplotlib',
        'meshio',
        'numpy',
        'scikit-image>=0.17',
        'scipy',
        'Pillow',
        'Cython',
    ],
    license='MIT',
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
    ),
    extras_require={
        'neural': ['torch']
    },
    ext_modules=ext_modules,
)
