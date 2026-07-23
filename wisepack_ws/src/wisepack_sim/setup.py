from glob import glob

from setuptools import find_packages, setup

package_name = 'wisepack_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/dds', glob('dds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='High Performance Creators',
    maintainer_email='office@hpc.bg',
    description='WISEPACK simulated perception and scenario publishing nodes',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'perception_sim = wisepack_sim.perception_sim:main',
        ],
    },
)
