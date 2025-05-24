from setuptools import setup, find_packages

setup(name='mesa_cfdna',
      version='0.3.0',
      description='Multimodal Epigenetic Sequencing Analysis (MESA) is a flexible and sensitive method of capturing and integrating multimodal epigenetic information of cfDNA using a single experimental assay.',
      url='https://github.com/ChaorongC/mesa_cfdna',
      author='Chaorong Chen',
      author_email='c.chen@uci.edu',
      license='BSD 3 clause',
      packages=find_packages(),
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      python_requires=">=3.6",
      keywords=['feature selection', 'machine learning', 'random forest', 'bioinformatic', 'multimodal', 'epigenetics','genomics','cancer detection'],
      install_requires=['numpy>=1.10.4',
                        'scikit-learn>=0.17.1',
                        'scipy>=0.17.0',
                        'Boruta',
                        'pandas',
                        'joblib',
                        'scipy'
                        ])
