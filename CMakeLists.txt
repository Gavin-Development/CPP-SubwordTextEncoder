cmake_minimum_required(VERSION 3.17)
project(TokenizerLibTest)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)

FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_library(TokenizerLib SubwordTextEncoder.cpp SubwordTextEncoder.h)
set_target_properties(TokenizerLib PROPERTIES PREFIX "")
target_link_libraries(TokenizerLib nlohmann_json::nlohmann_json pthread)

add_executable(TokenizerTest main.cpp)
target_link_libraries(TokenizerTest TokenizerLib)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/readme.txt ${CMAKE_CURRENT_BINARY_DIR}/readme.txt COPYONLY)