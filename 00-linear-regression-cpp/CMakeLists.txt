cmake_minimum_required (VERSION 3.10)

project( cRL )


# 设置编译指令支持 c++11 flags 
# Sets the compilation flags to use c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
# GLOB 命令搜索当前目录的所有 cpp, h 文件，添加到 SRCS 变量中。
file(GLOB SRCS *.cpp *.h)

# 添加一个 Utils 库引入 utils.cpp 源文件。
# Add a library called "Utils" which includes the source file "utils.cpp"
add_library(Utils utils.cpp)

# 添加可执行文件
add_executable(cRL ${SRCS})

# 将 Utils 库作为静态库引入 cRL 可执行程序中。
target_link_libraries(cRL Utils)


