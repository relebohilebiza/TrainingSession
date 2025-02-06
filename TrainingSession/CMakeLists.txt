# CMakeList.txt : CMake project for TrainingSession, include source and define
# project specific logic here.
#

project(TrainingSession)
set(CMAKE_CXX_STANDARD 20)

#Set c++11 for cpprestsdk
set(CMAKE_CXX_STANDARD 11 CACHE STRING "Set C++11 for cpprestsdk" FORCE)

# Include vcpkg toolchain for dependency management
set(CMAKE_TOOLCHAIN_FILE "C:/libs/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

set(CMAKE_PREFIX_PATH "C:/libs/vcpkg/installed/x64-windows")

# Set Torch directory if needed
set(Torch_DIR "C:/libs/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cpprestsdk CONFIG REQUIRED)

include_directories(${OpenCV_DIR})
# Add source to this project's executable.
add_executable (TrainingSession "TrainingSession.cpp" 
"TrainingSession.h"
)

target_link_libraries(TrainingSession ${TORCH_LIBRARIES} ${OpenCV_LIBS} cpprestsdk::cpprest )

# Copy data directory to the build directory
#file(COPY "data" DESTINATION ${CMAKE_BINARY_DIR})
# Copy the model 
#file(COPY tradition_model.pt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_compile_features(TrainingSession PRIVATE cxx_std_20)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET TrainingSession PROPERTY CXX_STANDARD 20)
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  file(GLOB OPENCV_DLLS "${OpenCV_DIR}/../bin/*.dll")  # Adjust the path as necessary
  add_custom_command(TARGET TrainingSession
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:TrainingSession>)
endif (MSVC)
# TODO: Add tests and install targets if needed.