import json
from datetime import datetime, timedelta

meals = [
    ("breakfast", ["eggs", "toast", "oatmeal", "pancakes", "yogurt", "cereal", "avocado toast", "granola"]),
    ("lunch", ["salad", "sandwich", "pasta", "rice", "chicken", "sushi", "burger", "wrap"]),
    ("dinner", ["steak", "pasta", "curry", "soup", "stir fry", "tacos", "pizza", "fish"]),
    ("snack", ["banana", "apple", "cookies", "chips", "granola bar", "protein bar", "nuts", "yogurt"]),
    ("drink", ["coffee", "tea", "juice", "soda", "smoothie", "water"]),
    ("dessert", ["ice cream", "cake", "cookie", "pie"]),
]

notes = []
start = datetime.strptime("06:00", "%H:%M")
for i in range(200):
    # time increments by 7 minutes each entry
    t = (start + timedelta(minutes=7 * i)).time()
    time_str = t.strftime("%H:%MUTC")

    meal_type, pool = meals[i % len(meals)]
    # pick 1-3 foods deterministically
    foods = [pool[(i + j) % len(pool)] for j in range(1 + (i % 3))]

    # create a natural note text
    if meal_type in ("breakfast", "lunch", "dinner"):
        note = f"{time_str} - Made {meal_type} ({', '.join(foods)}), ate shortly after."
        output = [
            {"timestamp": time_str, "label": f"{meal_type}", "food": foods}
        ]
    elif meal_type == "snack":
        note = f"{time_str} - Had a quick snack ({', '.join(foods)})."
        output = [
            {"timestamp": time_str, "label": "snack", "food": foods}
        ]
    elif meal_type == "drink":
        note = f"{time_str} - Drank {', '.join(foods)}."
        output = [
            {"timestamp": time_str, "label": "drink", "food": foods}
        ]
    else:
        note = f"{time_str} - Had dessert ({', '.join(foods)})."
        output = [
            {"timestamp": time_str, "label": "dessert", "food": foods}
        ]

    notes.append({"note": note, "output": output})

with open("/srv/ai-server/data/training_data.json", "w") as f:
    json.dump(notes, f, indent=2)

print(f"Wrote {len(notes)} entries to /srv/ai-server/data/training_data.json")
