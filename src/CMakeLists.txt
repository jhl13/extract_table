add_executable(
extract_table_node
main.cpp
)

target_link_libraries( extract_table_node ${OpenCV_LIBS} Threads::Threads)
