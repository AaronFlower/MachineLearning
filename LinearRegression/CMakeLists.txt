cmake_minimum_required ( VERSION 3.10 )

project( LinearReg )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )


add_executable(main main.cpp Utils.cpp)

