# Compile module to dv runtime
add_library(m_reclusive_event_denoisor SHARED module_dv.cpp)
set_target_properties(m_reclusive_event_denoisor PROPERTIES PREFIX "K")
target_link_libraries(m_reclusive_event_denoisor PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
install(TARGETS m_reclusive_event_denoisor DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Convert module to python package
pybind11_add_module(reclusive_event_denoisor module_py.cpp)
target_link_libraries(reclusive_event_denoisor PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
