from setuptools import setup

setup(
    name='bias-response-curve',
    version='0.1',
    py_modules=['bias_response_curve'],
    entry_points={
        'console_scripts': [
            'experiment = bias_response_curve:main',
        ],
    },
    install_requires=[],  # dependencies handled by requirements.txt
)
