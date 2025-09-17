from setuptools import find_packages, setup

package_name = 'voice_command'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ron',
    maintainer_email='ron@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            '1_get_command = voice_command.1_get_command:main',
            '2_command_to_nav2 = voice_command.2_command_to_nav2:main',
        ],
    },
)
