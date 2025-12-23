from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="ivc-circuit-detector",
      version="0.2.0",
      description="Detect circuit schema related to I-V curve",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/EPC-MSU/ivc-circuit-detector",
      author="EPC MSU",
      author_email="zap@physlab.ru",
      license="CC0-1.0",
      packages=["circuit_detector"],
      python_requires=">=3.6",
      install_requires=[
          'numpy==1.18.1; python_version=="3.6"',
          'numpy; python_version>"3.6"',
          "scikit-learn==0.24.2",
          "scipy",
          "epcore @ git+https://github.com/EPC-MSU/epcore@dev-0.2#egg=epcore"
      ],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CC0 License",
            "Operating System :: OS Independent",
      ],
      zip_safe=False)
