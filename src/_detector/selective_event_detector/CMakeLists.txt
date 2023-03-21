# Compile module to dv runtime
add_library(m_selective_event_detector SHARED module_dv.cpp)
set_target_properties(m_selective_event_detector PROPERTIES PREFIX "K")
target_link_libraries(m_selective_event_detector PUBLIC ${OpenCV_LIBRARIES} dv::sdk pybind11::module)
install(TARGETS m_selective_event_detector DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Convert module to python package
pybind11_add_module(selective_event_detector module_py.cpp)
target_link_libraries(selective_event_detector PUBLIC ${OpenCV_LIBRARIES} dv::sdk pybind11::module)
