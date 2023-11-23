cmake_minimum_required(VERSION 3.15)

project(optimTest LANGUAGES Fortran)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -Minfo=all -cuda -acc")
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -gpu=cc61,debug,lineinfo,ptxinfo,autocompare")
set(CMAKE_Fortran_FLAGS_RELEASE "-gopt -O1 -gpu=cc61,lineinfo,fma,lto,ptxinfo,unroll")

file(GLOB_RECURSE sources src/*.f90)
add_executable(optimTest ${sources})