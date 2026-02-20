cmake_minimum_required(VERSION 3.22)
project(ZMQ_Comm)

set(CMAKE_CXX_STANDARD 17)

# Find PythonLibs to get the include directories
# Note: "sudo apt install python3.10-dev" if not found
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})

# Find the pybind11 package
# Note: "sudo apt-get install pybind11-dev" if not found
find_package(pybind11 REQUIRED)

# Find the PyTorch package
# Note: Prefer the active Python environment (e.g., conda env) over hardcoded paths.
# We query torch.utils.cmake_prefix_path from the Python interpreter CMake found.
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT "${TORCH_CMAKE_PREFIX_PATH}" STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
endif()
find_package(Torch REQUIRED)

# Find libtorch_python.so, this is not included in the torch package by default
find_library(torch_python_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Find cppzmq (ZeroMQ C++ binding)
find_package(cppzmq REQUIRED)

# --------------------------------------------- Targets --------------------------------------------- #

# for verification of communication
add_executable(packed_server tests/test_packed_server.cpp)
add_executable(packed_client tests/test_packed_client.cpp)
target_link_libraries(packed_server cppzmq)
target_link_libraries(packed_client cppzmq)

# for verification of msg encoding & decoding
add_executable(test_msg tests/test_msg.cpp)
target_link_libraries(test_msg cppzmq)

# llm worker
pybind11_add_module(llm_worker MODULE src/sys_worker_api.cpp)
target_include_directories(llm_worker PRIVATE ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(llm_worker PRIVATE cppzmq)
target_link_libraries(llm_worker PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(llm_worker PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(llm_worker PRIVATE ${torch_python_LIBRARY})
target_link_libraries(llm_worker PRIVATE pybind11::pybind11)
target_link_libraries(llm_worker PRIVATE pybind11::embed)

# llm host
pybind11_add_module(llm_host MODULE src/sys_host_api.cpp)
target_include_directories(llm_host PRIVATE ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(llm_host PRIVATE cppzmq)
target_link_libraries(llm_host PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(llm_host PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(llm_host PRIVATE ${torch_python_LIBRARY})
target_link_libraries(llm_host PRIVATE pybind11::pybind11)
target_link_libraries(llm_host PRIVATE pybind11::embed)
