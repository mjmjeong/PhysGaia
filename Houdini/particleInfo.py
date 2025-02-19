import hou
import os
import json

# Get the node containing particle data (change the path to match your node)
flipsolver_node = hou.node("/obj/geo1/flipsolver1")
static_object_node = hou.node("/obj/geo1/matchsize1")

# Output directory
output_dir = "/Users/kimgunhee/Documents/24-2/랩인턴/HoudiniFiles/Fluid/Trajectory"
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

# Frame range 설정
start_frame = 1
end_frame = 1

for i, frame in enumerate(range(start_frame, end_frame + 1), start=1):
    hou.setFrame(frame)  # Set current Houdini frame
    
    particle_data = []
    last_pid = 0
    
    # Process particle data from flipsolver_node.
    particle_geo = flipsolver_node.geometry()
    
    for point in particle_geo.points():
        pid = point.number()
        last_pid = pid
        pos = point.position()  # (x, y, z)
        
        particle_data.append({
            "id": int(pid),
            "position": [pos[0], pos[1], pos[2]],
        })
    
    # Process static object data from static_object_node.
    static_geo = static_object_node.geometry()
    
    for point in static_geo.points():
        pid = point.number() + last_pid  
        pos = point.position()
        
        particle_data.append({
            "id": int(pid),
            "position": [pos[0], pos[1], pos[2]],
        })
    
    # Save JSON file with zero-padded frame number.
    json_file = os.path.join(output_dir, f"particles_frame_{frame:04d}.json")
    try:
        with open(json_file, "w") as f:
            json.dump(particle_data, f, indent=4)
        print(f"[{i}/{end_frame - start_frame + 1}] Saved JSON: {json_file}")
    except IOError as e:
        print(f"파일 저장 중 오류 발생: {e}")