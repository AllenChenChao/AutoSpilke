# 简单HLS流程
open_project simple_processor_proj -reset
add_files simple_processor.cpp
add_files -tb test_simple_processor.cpp
set_top simple_processor

open_solution solution1 -reset
set_part xc7z020clg400-1
create_clock -period 10

# 运行HLS流程
csim_design
csynth_design
cosim_design -rtl verilog
export_design -format ip_catalog

puts "✅ HLS Complete!"
exit