from glob import glob
from setuptools import setup

package_name = "intrinsic_motivation_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/urdf", glob("urdf/*.xacro")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/gui", glob("gui/*.config")),
    ],
    install_requires=["setuptools", "intrinsic_core"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Empowerment-driven intrinsic motivation for a UR5 in Gazebo.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "empowerment_node = intrinsic_motivation_ros.empowerment_node:main",
            "greedy_climber = intrinsic_motivation_ros.greedy_climber:main",
            "drifter_driver = intrinsic_motivation_ros.drifter_driver:main",
            "detector_eval = intrinsic_motivation_ros.detector_eval:main",
            "grasp_climber = intrinsic_motivation_ros.grasp_climber:main",
            "empowerment_node_planar = intrinsic_motivation_ros.empowerment_node_planar:main",
            "greedy_climber_planar = intrinsic_motivation_ros.greedy_climber_planar:main",
        ],
    },
)
