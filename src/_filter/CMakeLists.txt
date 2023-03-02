# find all source files in a directory
aux_source_directory(${PROJECT_SOURCE_DIR}/_filter DIR_SRCS)

# convert to .so file
pybind11_add_module(eventdenoisor ${DIR_SRCS})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# link to open_cblas
target_link_libraries(eventdenoisor PRIVATE ${BLAS_LIBRARIES})
