from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='ficbot',
   version='1.0',
   description='An AI-powered Fan Fiction Writing Assistant.',
   license="BSD-3-Clause License",
   #long_description=long_description,
   author='Kirill Nikolaev',
   author_email='kir.nikolaev.7@gmail.com',
   url="https://github.com/Pythonimous/fic-bot",
   packages=find_packages(include=['ficbot', 'ficbot.*']),
   install_requires=['wheel', 'setuptools', 'numpy', 'cython',
                     'Pillow', 'ImageHash',
                     'requests', 'bs4', 'beautifulsoup4', 'selenium',
                     'tqdm',  'pandas', 'tensorflow-cpu']
)