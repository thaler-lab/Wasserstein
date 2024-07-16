cmake_minimum_required(VERSION 3.20...3.29)
# 3.20 is the minimum requirement for LLVM
# https://www.llvm.org/docs/CMake.html#quick-start

project(
        ${SKBUILD_PROJECT_NAME}
	VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES CXX)

set(
	CMAKE_CXX_STANDARD
	14
	CACHE STRING "The C++ standard to compile with, 14+")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
	OpenMP
	REQUIRED)

find_package(
	Python
	COMPONENTS Interpreter Development.Module NumPy
	REQUIRED)

find_package(SWIG REQUIRED)
include(UseSWIG)

message(STATUS "Working directory is ${CMAKE_CURRENT_SOURCE_DIR}")

set_property(SOURCE "wasserstein/swig/wasserstein.i" PROPERTY CPLUSPLUS ON)
set_property(SOURCE "wasserstein/swig/wasserstein.i" PROPERTY SWIG_MODULE_NAME _wasserstein_omp)
swig_add_library(
	swig_wasserstein
        TYPE MODULE
	LANGUAGE python
	SOURCES "wasserstein/swig/wasserstein.i")
set_property(TARGET swig_wasserstein PROPERTY SWIG_COMPILE_OPTIONS -fastproxy -w511 -keyword)
set_property(TARGET swig_wasserstein PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)
target_include_directories(swig_wasserstein PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(swig_wasserstein PUBLIC -ffast-math -fPIC)

message(STATUS "Working directory is ${CMAKE_CURRENT_SOURCE_DIR}")

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
	OpenMP::OpenMP_CXX
	Python::Module
	Python::NumPy)

install(TARGETS swig_wasserstein DESTINATION .)
