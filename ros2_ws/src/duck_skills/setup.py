from setuptools import setup
package_name = "duck_skills"
setup(
    name=package_name, version="0.1.0", packages=[package_name],
    data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]),
                ("share/" + package_name, ["package.xml"])],
    install_requires=["setuptools"], zip_safe=True,
    entry_points={"console_scripts": ["skills_node = duck_skills.skills_node:main", "walk_to_node = duck_skills.walk_to_node:main", "push_ball_node = duck_skills.push_ball_node:main"]},
)
