project(KVCache LANGUAGES CUDA)

add_library(kvcache INTERFACE)
target_include_directories(kvcache INTERFACE include)
target_link_libraries(kvcache INTERFACE lslab)
