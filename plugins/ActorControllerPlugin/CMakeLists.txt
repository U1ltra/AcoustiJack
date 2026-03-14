cmake_minimum_required(VERSION 3.28.3 FATAL_ERROR)

# Find Gazebo packages
find_package(gz-cmake3 REQUIRED)

# Set project name
project(ActorControllerPlugin)

set(CMAKE_CXX_STANDARD 17) 

# Find MAVSDK
find_package(MAVSDK REQUIRED)

# Find the required Gazebo components (for Gazebo Sim 8.x)
gz_find_package(gz-plugin2 REQUIRED COMPONENTS register)
set(GZ_PLUGIN_VER ${gz-plugin2_VERSION_MAJOR})

# Optional: You can specify the version of gz-common if needed
find_package(gz-common5 REQUIRED)  
set(GZ_COMMON_VER ${gz-common5_VERSION_MAJOR})

include_directories(${rclcpp_INCLUDE_DIRS})
include_directories(${geometry_msgs_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${MAVSDK_INCLUDE_DIRS})

# Find other necessary Gazebo components
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(rclcpp REQUIRED)        
find_package(geometry_msgs REQUIRED)  
find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)
 
find_package(ros_gz_sim REQUIRED)
find_package(ros_gz REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-math7 REQUIRED)

# Define the plugin sources and add the shared library
add_library(${PROJECT_NAME} SHARED src/actor_controller_plugin.cc)

# Set C++ standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

link_directories(${GAZEBO_LIBRARY_DIRS})

# Link necessary libraries
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    gz-common${GZ_COMMON_VER}::gz-common${GZ_COMMON_VER}
    gz-plugin${GZ_PLUGIN_VER}::gz-plugin${GZ_PLUGIN_VER}
    gz-sim8::gz-sim8
    gz-transport13::gz-transport13
    rclcpp::rclcpp
    ${ros_gz_LIBRARIES}
    ${ros_gz_sim_LIBRARIES}
    ${ros_gz_bridge_LIBRARIES}
    ${gz-msgs10_LIBRARIES}
    ${GAZEBO_LIBRARIES}
    ${rclcpp_LIBRARIES}
    ${gazebo_ros_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${std_msgs_LIBRARIES}
    MAVSDK::mavsdk                   # Core MAVSDK library
)

# Include directories for the plugin
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${MAVSDK_INCLUDE_DIRS}  # Add MAVSDK include directories
)

# Install the plugin library
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-export
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Install CMake configuration
install(EXPORT ${PROJECT_NAME}-export
  FILE ${PROJECT_NAME}Config.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)