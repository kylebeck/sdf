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
    version='0.2.0',
    description='Generate 3D meshes from signed distance functions with GPU acceleration and dual contouring.',
    url='https://github.com/kylebeck/sdf',
    author='Michael Fogleman',
    author_email='michael.fogleman@gmail.com',
    maintainer='Kyle Beck',
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
    extras_require={
        'gpu': ['torch'],
        'taichi': ['taichi'],
        'all': ['torch', 'taichi'],
    },
    license='MIT',
    classifiers=(
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
    ),

    ext_modules=ext_modules,
)
