import os
import trimesh
from trimesh.visual import ColorVisuals

source_folder = "pix2pose_datasets/itodd/models"
target_folder = "pix2pose_datasets/itodd/models_converted"

def add_color_to_ply(source_path, target_path):
    mesh = trimesh.load(source_path)
    vertex_colors = [(128, 128, 128, 255)] * mesh.vertices.shape[0]
    
    mesh.visual = ColorVisuals(vertex_colors=vertex_colors)
    
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    mesh.export(target_path, file_type='ply')

# Process each ply file in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".ply"):
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        add_color_to_ply(source_path, target_path)

print("RGB color and alpha added to vertices using trimesh, files saved to the target folder.")