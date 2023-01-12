cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)


find_package(Python3 COMPONENTS Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
include_directories(.)

add_subdirectory(k-armed-bandit)
add_subdirectory(jacks_rental_car)
add_subdirectory(gambler)
add_subdirectory(blackjack)
