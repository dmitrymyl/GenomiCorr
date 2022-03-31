from setuptools import setup


setup(name='GenomiCorr',
      version='0.2.1',
      description='Genomic regions correlation analysis toolkit.',
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