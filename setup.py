from setuptools import setup
from glob import glob

package_name = "ros_voice"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="leidianzei7",
    maintainer_email="lei15988006987@gmail.com",
    description="语音交互 ROS 2 节点集（voice_node + brain_node + control_node）",
    license="MIT",
    entry_points={
        "console_scripts": [
            "voice_node   = ros_voice.voice_node:main",
            "brain_node   = ros_voice.brain_node:main",
            "control_node = ros_voice.control_node:main",
        ],
    },
)
