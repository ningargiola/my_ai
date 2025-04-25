from setuptools import setup, find_packages

setup(
    name="personal-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ollama>=0.1.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.4.2",
        "pydantic-settings>=2.0.3",
        "sqlalchemy>=2.0.23",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "python-multipart>=0.0.6",
        "pytest>=7.4.0",
        "httpx>=0.24.0",
    ],
) 