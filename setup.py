from setuptools import setup


setup(
    name="ehr_diagnosis_agent",
    version="0.0.1",
    install_requires=["gymnasium==0.28.1", "transformers==4.37.2", "accelerate==0.21.0", "pandas==2.0.1",
                      "sentencepiece==0.1.99", "sentence-transformers==2.2.2", "wandb==0.15.3", "omegaconf==2.3.0"],
)
