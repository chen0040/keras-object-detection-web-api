from setuptools import setup

setup(
    name='keras_object_detection',
    packages=['keras_object_detection'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)