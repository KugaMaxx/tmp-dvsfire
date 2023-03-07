# find all source files in a directory
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} DIR_SRCS)

# convert to .so file
# pybind11_add_module(eventdenoisor ${DIR_SRCS})
add_library(eventdenoisor SHARED reclusive_event_denoisor.cpp)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# link to open_cblas
target_link_libraries(eventdenoisor PRIVATE ${BLAS_LIBRARIES} dv::sdk pybind11::module)

# add_library(test SHARED reclusive_event_denoisor.cpp)
# target_link_libraries(test PRIVATE ${BLAS_LIBRARIES} dv::sdk)
install(TARGETS eventdenoisor DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
