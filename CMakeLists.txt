cmake_minimum_required(VERSION 3.17)
project(TokenizerLibTest)

set(CMAKE_CXX_STANDARD 14)

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
    target_link_libraries(${TESTNAME} gtest gtest_main rt pthread ${LIBRARIES})

    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

include(FetchContent)

FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_library(TokenizerLib SubwordTextEncoder.cpp Tokenizers.h)
set_target_properties(TokenizerLib PROPERTIES PREFIX "")
target_link_libraries(TokenizerLib nlohmann_json::nlohmann_json pthread)

package_add_test(runUnitTests unittests.cpp TokenizerLib)

add_executable(TokenizerTest main.cpp)
target_link_libraries(TokenizerTest TokenizerLib)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/readme.txt ${CMAKE_CURRENT_BINARY_DIR}/readme.txt COPYONLY)