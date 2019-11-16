cmake_minimum_required(VERSION 3.0)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

set (SOURCES main.cpp
             NetCdfWriter.cpp
             OutputInterface.h
             NetCdfWriter.h
             )

set (KERNELS test.cl)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_custom_command(
    OUTPUT  schema.h
    COMMAND ${PARENT_DIR}/Workflow/scripts/generateSchema.py -d ${PARENT_DIR}/description.yaml -f ${CMAKE_CURRENT_BINARY_DIR}/schema.h
    DEPENDS ${PARENT_DIR}/description.yaml
)

add_custom_command(
    OUTPUT  test.h
    COMMAND ${PARENT_DIR}/CLDE/util/clToHeader.py -i ${CMAKE_CURRENT_SOURCE_DIR}/test.cl -o ${CMAKE_CURRENT_BINARY_DIR}/test.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test.cl
)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
include_directories (~/include)
include_directories (../CLDE/include)


add_subdirectory(../CLDE/src ${CMAKE_CURRENT_BINARY_DIR}/clde)
add_executable(${PROJECT_NAME}
    ${SOURCES}
    schema.h
    test.h
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_link_libraries (${PROJECT_NAME} ~/lib/libyaml-cpp.so
                                        clde
                                        netcdf_c++4
                                        netcdf)



