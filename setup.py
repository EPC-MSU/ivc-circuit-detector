from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='ivc-circuit-detector',
      version='0.0.1',
      description='Detect circuit schema related to I-V curve',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/EPC-MSU/ivc-circuit-detector',
      author='EPC MSU',
      author_email='marakulin_ap@physlab.ru',
      license='CC0-1.0',
      packages=['circuit_detector'],
      install_requires=[
            '',  # YOUR DEPENDENCIES ARE HERE
      ],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CC0 License",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      zip_safe=False)
