from setuptools import find_packages, setup

with open("requirements.txt") as fh:
    requirements = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

setup(
    name="mtp-surgical-ultrasound",
    version="0.2.0",
    packages=find_packages(include=["ReconstructionPipeline", "ReconstructionPipeline.*",
                                    "surgical_physics_bridge", "surgical_physics_bridge.*",
                                    "mtp_gym", "mtp_gym.*"]),
    python_requires=">=3.10",
    install_requires=requirements,
)
