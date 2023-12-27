from setuptools import setup, find_packages

setup(
    name='cleangpt',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.1",
        "beautifulsoup4",
    ],
    extras_require={
        "dev": [
            "pylint",
        ]
    },
    author_email='mkon@hey.com',
    description='Reimplementation of GPT model',
    url='https://github.com/MichaelKonobeev/cleangpt',
)
