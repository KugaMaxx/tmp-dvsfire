# Compile module to dv runtime
add_library(selective_detector SHARED module_dv.cpp)
set_target_properties(selective_detector PROPERTIES PREFIX "Kuga_")
target_link_libraries(selective_detector PUBLIC ${OpenCV_LIBRARIES} dv::sdk pybind11::module)
install(TARGETS selective_detector DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Convert module to python package
pybind11_add_module(event_detector module_py.cpp)
target_link_libraries(event_detector PUBLIC ${OpenCV_LIBRARIES} dv::sdk pybind11::module)
