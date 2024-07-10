cmake_minimum_required(VERSION 3.20...3.29)
# 3.20 is the minimum requirement for LLVM
# https://www.llvm.org/docs/CMake.html#quick-start

project(
        ${SKBUILD_PROJECT_NAME}
	VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES CXX)

add_library(wasserstein OBJECT "wasserstein/wasserstein.cpp")
target_include_directories(wasserstein PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(wasserstein PUBLIC -ffast-math)

find_package(
	OpenMP
	REQUIRED)

find_package(
	Python
	COMPONENTS Interpreter Development.Module NumPy
	REQUIRED)

include_directories(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
include(FindSWIG REQUIRED)
include(UseSWIG)

set_property(SOURCE "wasserstein/swig/wasserstein.i" PROPERTY CPLUSPLUS ON)
swig_add_library(
	swig_wasserstein
        TYPE MODULE
	LANGUAGE python
	SOURCES "wasserstein/swig/wasserstein.i" "wasserstein/wasserstein.cpp")
target_include_directories(swig_wasserstein PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/wasserstein)
set_property(TARGET "swig_wasserstein" PROPERTY SWIG_COMPILE_OPTIONS -fastproxy -w511 -keyword)


if(WIN32)
	set_property(TARGET swig_wasserstein
		PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
	set_property(TARGET swig_wasserstein
		PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

target_link_libraries(
	swig_wasserstein
	PUBLIC
	wasserstein
	OpenMP::OpenMP_CXX
	Python::Module
	Python::NumPy)

install(TARGETS swig_wasserstein DESTINATION .)
