
import sys
from os import path

from setuptools import setup, find_packages

try:
    from mytools import __about__ as about
except ImportError:
    
    sys.path.append("mytools")
    import __about__ as about

PATH_HERE = path.abspath(path.dirname(__file__))

with open(path.join(PATH_HERE, 'requirements.txt'), encoding='utf-8') as fp:
    requirements = [rq.rstrip() for rq in fp.readlines() if not rq.startswith('#')]


setup(
    name='mytools',
    version=about.__version__,
    url=about.__homepage__,
    author=about.__author__,
    author_email=about.__author_email__,
    description=about.__doc__,
    packages=find_packages(),
    keywords='shap,fi,pipeline',
    install_requires=requirements,
    python_requires='>=3.6',
    include_package_data=True,
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',

        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)
