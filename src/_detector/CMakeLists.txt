# find all source files in a directory
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} DIR_SRCS)

# convert to .so file
pybind11_add_module(eventdetector ${DIR_SRCS})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# link to opencv
target_link_libraries(eventdetector PRIVATE ${OpenCV_LIBRARIES})
