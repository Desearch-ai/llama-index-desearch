from setuptools import setup, find_packages

setup(
    name="llama-index-desearch",  # Replace with your desired package name
    version="0.0.1",  # Initial version
    author="Desearch",  # Replace with your name
    author_email="your-email@example.com",  # Replace with your email
    description="Llama-index integration with Desearch API for search and data-fetching tools.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Desearch-ai/llama-index-desearch",  # Replace with your GitHub repo URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "llama-index>=0.12.34",
        "desearch-py>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
