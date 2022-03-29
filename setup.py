from setuptools import setup


setup(name='genomicorr',
      version='0.1.0',
      description='Genomic regions correlation analysis tools.',
      url='http://github.com/dmitrymyl/GenomiCorr',
      author='Dmitry Mylarshchikov',
      author_email='dmitrymyl@gmail.com',
      license='MIT',
      packages=['genomicorr'],
      install_requires=['numpy',
                        'pandas',
                        'bioframe',
                        'scipy',
                        'statsmodels'],
      python_requires='>=3.7',
    #   entry_points={'console_scripts':
    #                 ['ortho2align=ortho2align.cli_scripts:ortho2align']},
      zip_safe=True)