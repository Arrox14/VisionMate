from typing import List


def generate_scene_description(detected_objects: list[str]) -> str:
    """
    Generate a natural-language description for the objects detected in the scene.

    Args:
        detected_objects: A list of object labels detected by the system.

    Returns:
        A human-readable sentence describing the objects.

    Examples:
        >>> generate_scene_description(["person"])
        'There is a person in front of you.'
        >>> generate_scene_description(["person", "phone"])
        'There is a person and a phone in front of you.'
        >>> generate_scene_description(["person", "person", "phone"])
        'There is a person and a phone in front of you.'
    """
    unique_objects = list(dict.fromkeys(detected_objects))

    if not unique_objects:
        return "There is nothing in front of you."

    if len(unique_objects) == 1:
        return f"There is a {unique_objects[0]} in front of you."

    if len(unique_objects) == 2:
        first, second = unique_objects
        return f"There is a {first} and a {second} in front of you."

    formatted = ", ".join(f"a {obj}" for obj in unique_objects[:-1])
    return f"There is {formatted}, and a {unique_objects[-1]} nearby."
