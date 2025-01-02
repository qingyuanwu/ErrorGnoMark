from setuptools import setup, find_packages

# Function to read the dependencies from the requirements.txt file
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return f.read().splitlines()

# Function to read the long description (usually from README.md)
def read_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name='ErrorGnoMark',  # Package name
    version='0.1.0',  # Initial version number
    author='Chai Xudan, Wang Jingbo, Xiaoxiao',  # Author/organization name
    author_email='chaixd@baqis.ac.cn',  # Contact email
    url='https://github.com/BAQIS-Quantum/ErrorGnoMark',  # Project repository URL
    description='A Toolkit for Error Diagnosis and Benchmarking for Quantum Chip',  # Short description
    long_description=read_long_description(),  # Detailed description from README.md
    long_description_content_type='text/markdown',  # Markdown format for the long description
    keywords=["errorgnomark", "quantum benchmarking", "quantum computing", "quantum operating system"],  # Keywords
    packages=find_packages(),  # Automatically discover all sub-packages
    install_requires=read_requirements(),  # Dependencies from requirements.txt
    python_requires='>=3.10',  # Minimum Python version
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    license="Apache-2.0 License"  # License type
)
