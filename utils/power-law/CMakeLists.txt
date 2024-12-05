cmake_minimum_required(VERSION 3.14)

project(power-law CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

find_package(Boost 1.70 COMPONENTS system program_options)

file(GLOB sources *.cc)
file(GLOB headers *.hh)

set(SOURCE_FILES ${sources} ${headers})
add_executable(pl ${SOURCE_FILES})
target_link_libraries(pl PUBLIC Boost::program_options)
target_compile_options(pl PUBLIC -Wall -Wpedantic)
set_target_properties(pl PROPERTIES CXX_STANDARD 17)