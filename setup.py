from setuptools import setup, find_packages


setup(
    name='lexibank_sabor',
    version='0.1.0.dev0',
    description='',
    author="John Miller and Johann-Mattis List",
    author_email='jemiller@pucp.edu.pe',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='',
    license='MIT',
    url='https://github.com/lexibank/sabor',
    py_modules=['lexibank_sabor'],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'lexibank.dataset': [
            'sabor=lexibank_sabor:Dataset',
        ],
        'cldfbench.commands': [
            'sabor=saborcommands',
        ],
    },
    platforms='any',
    python_requires='>=3.7',
    install_requires=[
        'attrs>=18.2',
        'cldfbench>=1.7.2',
        'cldfcatalog>=1.3',
        'clldutils>=3.12.0',
        'cltoolkit>=0.1.1',
        'cldfviz>=0.3.0',
        'cldfzenodo',
        'GitPython>=3.1.27'
        'csvw>=3.1.3',
        'lingpy>=2.6.8',
        'numpy>=1.22.4',
        'pycldf>=1.26.1',
        'pyclts>=3.1',
        'pylexibank>=3.3.0',
        'scikit_learn>=1.1.1',
        'tabulate>=0.8.9',
        'uritemplate>=4.1.1',
    ],
    extras_require={
        'dev': ['flake8', 'wheel', 'twine'],
        'test': [
            'pytest>=6',
            'pytest-mock',
            'pytest-cov',
            'pytest-cldf',
            'coverage',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
