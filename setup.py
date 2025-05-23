from setuptools import setup, find_packages
import versioneer

setup(
    name='mesa-cfdna',  # Use hyphens for PyPI package names
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Multimodal Epigenetic Sequencing Analysis (MESA) is a flexible and sensitive method of capturing and integrating multimodal epigenetic information of cfDNA using a single experimental assay.',
    url='https://github.com/ChaorongC/mesa_cfdna',  # Fixed URL to match repo name
    author='Chaorong Chen',
    author_email='c.chen@uci.edu',
    license='BSD 3 clause',
    packages=find_packages(),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    python_requires=">=3.7",  # Updated minimum Python version
    keywords=['feature selection', 'machine learning', 'random forest', 'bioinformatics', 'multimodal', 'epigenetics', 'genomics', 'cancer detection'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'numpy>=1.19.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'boruta>=0.3',  # Lowercase package name
        'pandas>=1.3.0',
        'joblib>=1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
        ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/ChaorongC/mesa_cfdna/issues',
        'Source': 'https://github.com/ChaorongC/mesa_cfdna',
        'Documentation': 'https://github.com/ChaorongC/mesa_cfdna/blob/main/README.md',
    },
    include_package_data=True,
    zip_safe=False,
)