cmake_minimum_required(VERSION 3.20...3.29)
# 3.20 is the minimum requirement for LLVM
# https://www.llvm.org/docs/CMake.html#quick-start

project(
        ${SKBUILD_PROJECT_NAME}
	VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES CXX)

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

add_library(wasserstein OBJECT "wasserstein/wasserstein.cpp")
target_include_directories(wasserstein PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(wasserstein PUBLIC -ffast-math)
set_property(TARGET wasserstein PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(wasserstein PUBLIC Python::Module Python::NumPy OpenMP::OpenMP_CXX)

set_property(SOURCE "wasserstein/swig/wasserstein.i" PROPERTY CPLUSPLUS ON)
swig_add_library(
	swig_wasserstein
        TYPE MODULE
	LANGUAGE python
	SOURCES "wasserstein/swig/wasserstein.i")
target_include_directories(wasserstein PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/wasserstein" "${CMAKE_CURRENT_SOURCE_DIR}/wasserstein/internal")
set_property(TARGET swig_wasserstein PROPERTY SWIG_COMPILE_OPTIONS -fastproxy -w511 -keyword)
set_property(TARGET swig_wasserstein PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)
target_include_directories(swig_wasserstein PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/wasserstein" "${CMAKE_CURRENT_SOURCE_DIR}/wasserstein/internal")

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
	wasserstein
	OpenMP::OpenMP_CXX
	Python::Module
	Python::NumPy)

install(TARGETS swig_wasserstein DESTINATION .)
