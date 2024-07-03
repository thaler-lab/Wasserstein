cmake_minimum_required(VERSION 3.20...3.29)
# 3.20 is the minimum requirement for LLVM
# https://www.llvm.org/docs/CMake.html#quick-start

project(
        ${SKBUILD_PROJECT_NAME}
	VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES CXX)

add_library(wasserstein OBJECT "wasserstein/wasserstein.cpp")

find_package(
	OpenMP
	REQUIRED)

find_package(
	Python
	COMPONENTS Interpreter Development.Module
	REQUIRED)

include(FindSWIG)
include(UseSWIG)

swig_add_library(
	swig_wasserstein
        TYPE MODULE
	LANGUAGE python
	SOURCES "wasserstein/swig/wasserstein.i" "wasserstein/wasserstein.cpp")
if(WIN32)
	set_property(TARGET swig_wasserstein 
		PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
	set_property(TARGET swig_wasserstein
		PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

target_link_libraries(swig_wasserstein PUBLIC wasserstein OpenMP::OpenMP_CXX Python::Module Python::NumPy)

install(TARGETS swig_wasserstein DESTINATION .)
