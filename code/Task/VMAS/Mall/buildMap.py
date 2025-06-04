import os
PATH = os.path.dirname(os.path.abspath(__file__)) + "/map.txt"
txt = ""
def add_wall(corner_1, corner_2):
    global txt
    
    txt = txt + "{:.2f},{:.2f},{:.2f},{:.2f}\n".format(corner_1[0], corner_1[1], corner_2[0], corner_2[1])

def add_room(corner_1, corner_2, open=""):
    x_min = min(corner_1[0], corner_2[0])
    x_max = max(corner_1[0], corner_2[0])
    y_min = min(corner_1[1], corner_2[1])
    y_max = max(corner_1[1], corner_2[1])

    for side in ["top", "down", "left", "right"]:
        if open == side:
            continue
        if side=="top":
            add_wall((x_min, y_max), (x_max, y_max-0.05))
        elif side=="down":
            add_wall((x_min, y_min), (x_max, y_min+0.05))
        elif side=="left":
            add_wall((x_min, y_min), (x_min+0.05, y_max))
        elif side=="right":
            add_wall((x_max, y_min), (x_max-0.05, y_max))

# # Vertical corridor
add_wall((-0.25, -0.9), (-0.2, -0.6))  
add_wall((0.25, -0.9), (0.2, -0.6)) 

add_wall((-0.25, -0.45), (-0.2, 0.1)) 
add_wall((0.25, -0.45), (0.2, 0.1))  

# Horizontal corridor
add_wall((-0.8, 0.1), (-0.6, 0.15)) 
add_wall((-0.45, 0.1), (-0.2, 0.15))

add_wall((0.8, 0.1), (0.6, 0.15)) 
add_wall((0.45, 0.1), (0.2, 0.15))

add_wall((-0.45, 0.5), (0.45, 0.45))
add_wall((-0.8, 0.5), (-0.6, 0.45)) 
add_wall((0.8, 0.5), (0.6, 0.45)) 

add_wall((-0.8, -0.3), (-0.75,0.45)) 
add_wall((0.8, -0.3), (0.75,0.45)) 

# Rooms
add_room((-0.8,-0.9), (-0.2, -0.3), open="right")
add_room((0.8,-0.9), (0.2, -0.3), open="left")

add_room((-0.8,0.5), (-0.2, 0.9), open="down")
add_room((0.8,0.5), (0.2, 0.9), open="down")

with open(PATH, "w") as f:
    f.write(txt)