import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="onnx-web",
    version="0.7.1",
    author="ssube",
    author_email="seansube@gmail.com",
    description="web UI for running ONNX models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ssube/onnx-web",
    keywords=[
        'onnx',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8,<3.11',
)
