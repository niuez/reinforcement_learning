add_executable(blackjack main.cpp)
add_executable(blackjack_rf main.cpp)
target_link_libraries(blackjack Python3::Python Python3::NumPy)
target_link_libraries(blackjack_rf Python3::Python Python3::NumPy)
