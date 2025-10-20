# 差分编码HLS流程
open_project differential_encoder_proj -reset
# open_project direct_prj -reset
add_files differential_encoder.cpp
add_files -tb test_differential_encoder.cpp
set_top differential_encoder

open_solution solution1 -reset
set_part xc7z020clg400-1
create_clock -period 10

# 运行HLS流程
csim_design
csynth_design
cosim_design -rtl verilog
export_design -format ip_catalog

puts "✅ Differential Encoder HLS Complete!"
exit