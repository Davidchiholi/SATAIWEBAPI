import json

from move_comparison import compare_positions



output_filename="C:/project/Sportatous/satfpyai/satfpyai/result.mp4" 
output_fullname="C:/project/Sportatous/satfpyai/satfpyai/result.mp4"
check_rate=60
sport="ABC"
show_window=False
combine_result=True
model='' 
equip='' 
model1=''
equip1=''
model2=''
equip2=''

videoinputUser = "C:/project/Sportatous/satfpyai/satfpyai/1.mp4"
videoinputBenchmark = "C:/project/Sportatous/satfpyai/satfpyai/2.mp4"

joints="25,0.3,26,0.3,27,0.2,28,0.2"

joints_dict = {}
joints_list = joints.split(",")
print(joints_list)
joints_dict["joint1"] = int(joints_list[0])
joints_dict["joint1_weighting"] = float(joints_list[1])
joints_dict["joint2"] = int(joints_list[2])
joints_dict["joint2_weighting"] = float(joints_list[3])
joints_dict["joint3"] = int(joints_list[4])
joints_dict["joint3_weighting"] = float(joints_list[5])
joints_dict["joint4"] = int(joints_list[6])
joints_dict["joint4_weighting"] = float(joints_list[7])   


result = compare_positions(videoinputBenchmark, videoinputUser, '', '', output_filename, output_fullname, '', check_rate, '', sport, show_window, combine_result, '', model, equip, model1, equip1, model2, equip2, joints_dict)

print(result)

