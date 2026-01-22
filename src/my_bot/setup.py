from setuptools import setup
import os
from glob import glob

package_name = 'my_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # Install marker file in the package index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        
        # Install Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        
        # Install Config (YAML) files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        
        # Install URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='LiDAR Scanning and Scoop Analysis Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'servo_driver = my_bot.servo_driver:main',
            'scan_accumulator = my_bot.scan_accumulator:main',
            'smart_wiper = my_bot.smart_wiper:main',
        ],
    },
)
