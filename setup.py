from setuptools import setup, find_packages

# Function to read the dependencies from the requirements.txt file
def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

# Function to read the long description (usually from README.md) for the PyPI page
def read_long_description():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

setup(
    name='ErrorGnoMark',  # The name of your package
    version='0.1.0',  # Version number, update as needed
    description='A Quantum Error Diagnosis and Benchmarking Toolkit',  # Short description of the project
    long_description=read_long_description(),  # Long description from README.md
    long_description_content_type='text/markdown',  # Format of the long description
    author='Chai Xudan, Wang Jingbo, Xiaoxiao ',  # Author of the project
    author_email='chaixd@baqis.ac.cn',  # Author email
    url='https://github.com/BAQIS-Quantum/ErrorGnoMark',  # URL to the project repository
    packages=find_packages(),  # Automatically find all sub-packages
    install_requires=read_requirements(),  # Read dependencies from requirements.txt
    classifiers=[  # Classifiers for PyPI
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Minimum Python version required
    include_package_data=True,  # Include additional files like README.md and LICENSE
)

