# Compile module to dv runtime
# add_library(reclusive_filter SHARED module_dv.cpp)
# set_target_properties(reclusive_filter PROPERTIES PREFIX "Kuga_")
# target_link_libraries(reclusive_filter PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
# install(TARGETS reclusive_filter DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Convert module to python package
pybind11_add_module(event_detector module_py.cpp)
target_link_libraries(event_detector PUBLIC ${OpenCV_LIBRARIES} dv::sdk pybind11::module)
