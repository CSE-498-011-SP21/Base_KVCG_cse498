project(KVStore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(kvstore INTERFACE)
target_include_directories(kvstore INTERFACE include)
target_link_libraries(kvstore INTERFACE multithreading)
target_link_libraries(kvstore INTERFACE pthread)
target_link_libraries(kvstore INTERFACE lslab)
target_link_libraries(kvstore INTERFACE kvcache)
target_link_libraries(kvstore INTERFACE TBB::tbb)