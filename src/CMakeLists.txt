file(GLOB SOURCES "*.cpp")
add_library(nnvmc SHARED ${SOURCES})
target_link_libraries(nnvmc "${MCI_LIBRARY_DIR}" "${NFM_LIBRARY_DIR}" "${VMC_LIBRARY_DIR}" "${QNETS_LIBRARY_DIR}" "${SANNIFA_LIBRARY_DIR}" "${GSL_LIBRARIES}" "${MPI_CXX_LIBRARIES}") # shared libs
add_library(nnvmc_static STATIC ${SOURCES})
target_link_libraries(nnvmc_static "${MCI_STATIC_LIBRARY_DIR}" "${NFM_STATIC_LIBRARY_DIR}" "${VMC_STATIC_LIBRARY_DIR}" "${QNETS_STATIC_LIBRARY_DIR}" "${SANNIFA_LIBRARY_DIR}" "${GSL_LIBRARIES}" "${MPI_CXX_LIBRARIES}") # static (+ some shared) libs
