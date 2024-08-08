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

# Linking to OpenMP will be outside the source tree, so include it in INSTALL_RPATH anyway
# Repair steps will bundle it as needed.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

find_package(
	OpenMP
	REQUIRED)

find_package(
	Python
	COMPONENTS Interpreter Development.Module NumPy
	REQUIRED)

set(UseSWIG_MODULE_VERSION 2)
find_package(SWIG REQUIRED)
include(UseSWIG)

message(STATUS "Working directory is ${CMAKE_CURRENT_SOURCE_DIR}")

set_property(SOURCE "wasserstein/swig/wasserstein.i" PROPERTY CPLUSPLUS ON)
set_property(SOURCE "wasserstein/swig/wasserstein.i" PROPERTY SWIG_MODULE_NAME wasserstein)
swig_add_library(
	wasserstein
        TYPE MODULE
	LANGUAGE python
	SOURCES "wasserstein/swig/wasserstein.i")
set_property(TARGET wasserstein PROPERTY SWIG_COMPILE_OPTIONS -fastproxy -w511 -keyword)
set_property(TARGET wasserstein PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)
target_include_directories(wasserstein PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(wasserstein PUBLIC -ffast-math -fPIC)

message(STATUS "Working directory is ${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
	set_property(TARGET wasserstein
		PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
	set_property(TARGET wasserstein
		PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

target_link_libraries(
	wasserstein
	PUBLIC
	OpenMP::OpenMP_CXX
	Python::Module
	Python::NumPy)
 
install(TARGETS wasserstein DESTINATION wasserstein/omp)

get_property(support_files TARGET wasserstein PROPERTY SWIG_SUPPORT_FILES)
message(STATUS "${support_files}")
install(FILES "${support_files}" DESTINATION wasserstein)
