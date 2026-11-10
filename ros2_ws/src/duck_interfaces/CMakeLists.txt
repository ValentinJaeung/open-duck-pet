cmake_minimum_required(VERSION 3.8)
project(duck_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObject.msg"
  "msg/DetectedObjectArray.msg"
  "action/Turn.action"
  "action/Walk.action"
  "action/LookAt.action"
  "action/Happy.action"
  "action/Sad.action"
  "action/WagHead.action"
  "action/Dance.action"
  "action/WalkTo.action"
  "action/PushBall.action"
  DEPENDENCIES geometry_msgs std_msgs
)
ament_package()
