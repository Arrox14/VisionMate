from smart_narrator import generate_smart_description

print(
    generate_smart_description(
        "Arro",
        ["person", "cell phone"]
    )
)

print(
    generate_smart_description(
        "Arro",
        []
    )
)

print(
    generate_smart_description(
        None,
        ["bottle"]
    )
)