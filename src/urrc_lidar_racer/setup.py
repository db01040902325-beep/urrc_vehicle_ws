from setuptools import setup
from glob import glob
import os

package_name = 'urrc_lidar_racer'

setup(
    name=package_name,
    version='0.0.2',

    packages=[package_name],

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.rviz') + glob('config/*.yaml')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='urrc',
    maintainer_email='user@example.com',

    description='LiDAR + camera autonomous racing controller with overtaking',
    license='MIT',

    entry_points={
        'console_scripts': [
            'lidar_racer = urrc_lidar_racer.lidar_racer:main',
            'camera_perception = urrc_lidar_racer.camera_perception:main',
            'overtake_fusion = urrc_lidar_racer.overtake_fusion:main',
            'race_stats = urrc_lidar_racer.race_stats:main',
            'lap_timer = urrc_lidar_racer.lap_timer:main',
            'birdseye_view = urrc_lidar_racer.birdseye_view:main',
        ],
    },
)
