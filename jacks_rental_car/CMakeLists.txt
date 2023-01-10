add_executable(jacks_rental_car main.cpp)
target_link_libraries(jacks_rental_car Python3::Python Python3::NumPy)

add_executable(extend_rental_car ext_main.cpp)
target_link_libraries(extend_rental_car Python3::Python Python3::NumPy)

add_executable(rental_car_plot plot.cpp)
target_link_libraries(rental_car_plot Python3::Python Python3::NumPy)
