cmake_minimum_required(VERSION 3.17)
project(TokenizerLib)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_PYTHON_VERSION 3.8.3)

find_package(Python 3.8 EXACT REQUIRED COMPONENTS Interpreter Development)

option(gtest_disable_pthreads "Disable uses of pthreads in gtest." ON)

### Start PyBind11 config
configure_file(CMakeLists-Pybind11.txt.in pybind11-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11-download)
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11-download)

# Add pybind11 directly
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/pybind11-src
        ${CMAKE_CURRENT_BINARY_DIR}/pybind11-build
        EXCLUDE_FROM_ALL)

### START OF CONFIGURING GTEST
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)
### END OF CONFIGURING GTEST

enable_testing()
# find_package(GTest REQUIRED)

macro(package_add_test TESTNAME FILES LIBRARIES)
    add_executable(${TESTNAME} ${FILES})
    # Link google test
    target_link_libraries(${TESTNAME} gtest gtest_main ${LIBRARIES})

    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

include(FetchContent)

FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
        GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Configure files first.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/readme.txt ${CMAKE_CURRENT_BINARY_DIR}/readme.txt COPYONLY)

# Library
add_library(TokenizerLib SubwordTextEncoder.cpp Tokenizers.h)
set_target_properties(TokenizerLib PROPERTIES PREFIX "")
target_link_libraries(TokenizerLib PUBLIC nlohmann_json::nlohmann_json)

# GoogleTest Executable
package_add_test(runUnitTests unittests.cpp TokenizerLib)

# Regular Executable test.
add_executable(TokenizerTest main.cpp)
target_link_libraries(TokenizerTest TokenizerLib)

pybind11_add_module(GavinTokenizers MODULE NO_EXTRAS SubwordTextEncoder.cpp Tokenizers.h)
target_link_libraries(GavinTokenizers PUBLIC nlohmann_json::nlohmann_json)
set_target_properties(GavinTokenizers PROPERTIES SUFFIX ".pyd")
