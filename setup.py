from setuptools import setup

setup(
    name='bias-response-curve',
    version='0.1',
    packages=['BRC_Experiment', 'BRC_Experiment.Modularized'],
    entry_points={
        'console_scripts': [
            'experiment = BRC_Experiment.Modularized.cli:main',
        ],
    },
    install_requires=[],  # dependencies handled by requirements.txt
)
