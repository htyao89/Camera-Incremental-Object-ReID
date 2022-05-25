from setuptools import setup, find_packages


setup(name='cior',
      version='1.0.0',
      description='Camera-Incremental Object ReID',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages())
