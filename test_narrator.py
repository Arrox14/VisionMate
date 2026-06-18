# test_narrator.py

from agent.scene_narrator import generate_scene_description

print(generate_scene_description([]))
print(generate_scene_description(["person"]))
print(generate_scene_description(["person", "phone"]))
print(generate_scene_description(["person", "person", "phone"]))
print(generate_scene_description(["person", "chair", "bottle"]))