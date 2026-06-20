def generate_smart_description(
    person_name: str | None,
    objects: list[str]
) -> str:

    objects = [obj for obj in objects if obj != "person"]

    if person_name and objects:
        if len(objects) == 1:
            return (
                f"{person_name} is in front of you "
                f"with a {objects[0]}."
            )

        return (
            f"{person_name} is in front of you "
            f"with {', '.join(objects[:-1])} "
            f"and {objects[-1]}."
        )

    if person_name:
        return f"{person_name} is in front of you."

    if objects:
        if len(objects) == 1:
            return f"There is a {objects[0]} in front of you."

        return (
            "There are "
            + ", ".join(objects[:-1])
            + " and "
            + objects[-1]
            + " in front of you."
        )

    return "Nothing important detected."