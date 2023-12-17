cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED PATHS libtorch/share/cmake/Torch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

this is just for my original test
add_executable(test_raw test_raw.cc)
target_link_libraries(test_raw "${TORCH_LIBRARIES}")
set_property(TARGET test_raw PROPERTY CXX_STANDARD 17)

add_executable(test test.cc)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 17)

# add_executable(train train.cpp)
# target_link_libraries(train "${TORCH_LIBRARIES}")
# set_property(TARGET train PROPERTY CXX_STANDARD 17)


# cannot use libtorchaudio, not supported
# add_executable(prepare prepare.cc)
# target_link_libraries(prepare PRIVATE "${TORCH_LIBRARIES}" libtorchaudio)
# set_property(TARGET prepare PROPERTY CXX_STANDARD 17)

