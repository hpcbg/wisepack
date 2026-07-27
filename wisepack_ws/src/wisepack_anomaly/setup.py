from glob import glob
from setuptools import find_packages, setup

package_name = 'wisepack_anomaly'
setup(
    name=package_name, version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='High Performance Creators', maintainer_email='office@hpc.bg',
    description='SIMULATED anomaly-integration demonstration (application-independent)',
    license='MIT', extras_require={'test': ['pytest']},
    entry_points={'console_scripts': [
        'anomaly_simulator = wisepack_anomaly.anomaly_simulator_node:main',
        'anomaly_adapter = wisepack_anomaly.anomaly_adapter_node:main',
    ]},
)
