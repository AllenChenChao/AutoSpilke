# 简单两参数HLS测试脚本

set PROJECT_NAME "simple_test"
set SOLUTION_NAME "solution1"
set TOP_FUNCTION "simple_add"
# set TOP_FUNCTION "simple_add"
set DEVICE "xc7z020-clg400-1"
set CLOCK_PERIOD 10

puts "Begin Simple HLS Test..."

# 创建项目
open_project $PROJECT_NAME -reset

# 添加文件
add_files "simple_test.cpp"
add_files "simple_test.h"
add_files -tb "simple_test_tb.cpp"

# 设置顶层函数
set_top $TOP_FUNCTION

# 创建解决方案
open_solution $SOLUTION_NAME -reset
set_part $DEVICE
create_clock -period $CLOCK_PERIOD -name default

# C仿真
puts "Run C Simulation..."
csim_design -clean

# C综合
puts "Run C Synthesis..."
csynth_design

# C/RTL协同仿真
puts "Run C/RTL Co-Simulation..."
cosim_design -rtl verilog -trace_level none

puts "Simple HLS Test Completed"
close_project
exit 0