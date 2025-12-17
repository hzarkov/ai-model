import json
from datetime import datetime, timedelta

# The 40 variant notes provided by the user
variant_notes = [
"07:50UTC - Made scrambled eggs (eggs, milk, butter), ate by 08:05UTC.",
"08:30UTC - Grabbed a coffee and muffin (blueberry), headed to meeting.",
"09:10UTC - Prepared lunch (turkey sandwich, avocado), ate at 09:30UTC.",
"10:00UTC - Took a 10min snack break, had an apple and almonds.",
"11:45UTC - Cooked pasta (penne, tomato sauce, basil), dinner ready at 12:20UTC.",
"12:15UTC - Ordered takeout (pad thai, tofu), delivered at 12:40UTC.",
"13:05UTC - Heated leftovers (rice and chicken), ate quickly.",
"14:20UTC - Made a smoothie (banana, spinach, yogurt), drank at 14:30UTC.",
"15:00UTC - Had a tea and two biscuits, back to work at 15:10UTC.",
"16:50UTC - Stopped for street food (taco with beef), finished by 17:05UTC.",
"17:30UTC - Prepared stir fry (shrimp, bell pepper, soy), ate at 18:00UTC.",
"18:10UTC - Baked bread (flour, yeast), cooled and ate with butter later.",
"19:00UTC - Made curry (lentils, coconut milk), started eating at 19:25UTC.",
"20:15UTC - Ordered pizza (margherita), shared with team at 20:40UTC.",
"21:00UTC - Late snack: yogurt and granola, felt full afterwards.",
"06:40UTC - Had breakfast on the go (bagel with cream cheese), arrived at office.",
"07:25UTC - Cooked omelette (eggs, cheese, spinach), ate at 07:40UTC.",
"09:45UTC - Grabbed a pastry (croissant) and espresso, quick break.",
"12:00UTC - Lunch prep took 15min, made tuna salad (tuna, mayo, celery).",
"12:35UTC - Ate lunch outside (sushi roll: salmon, avocado), nice weather.",
"13:50UTC - Snacked on carrot sticks and hummus, refreshed.",
"15:30UTC - Made protein shake (whey, banana), drank before gym.",
"16:00UTC - Finished preparing dinner (roast chicken, potatoes) — smells good.",
"17:10UTC - Ate dinner at 17:30UTC, dessert was fruit salad (apple, kiwi).",
"18:45UTC - Quick coffee break, had a biscotti with latte.",
"19:20UTC - Took 25 minutes to cook spaghetti (garlic, olive oil, chili).",
"20:05UTC - Ate leftovers (stew) and warmed bread, done by 20:25UTC.",
"21:30UTC - Craving: ice cream (vanilla) — had a small bowl.",
"22:10UTC - Made late sandwich (peanut butter & jam), quick snack before bed.",
"11:10UTC - Prepped brunch (eggs benedict, hollandaise), served at 11:40UTC.",
"12:50UTC - Ate quickly between calls: salad wrap (chicken, lettuce, tzatziki).",
"14:05UTC - Snack: cheese and crackers while reading.",
"15:55UTC - Coffee and a slice of cake (lemon), meeting buffered.",
"17:00UTC - Prepped stew (beef, carrots, potatoes) — slow cook for 2 hours.",
"18:40UTC - Dinner served at 19:00UTC, family joined for pizza.",
"08:15UTC - Toast with jam, morning routine — left at 08:35UTC.",
"09:00UTC - Made pancakes for kids (flour, milk, eggs), finished by 09:25UTC.",
"12:10UTC - Quick stir-fry lunch (tofu, broccoli, teriyaki), ate at desk.",
"16:20UTC - Snacked on mixed nuts and dark chocolate, energized.",
"18:55UTC - Had a small dessert (brownie) after dinner, then cleaned up."
]

# helper to extract food tokens inside parentheses or after colon
import re

def extract_foods(note):
    foods = []
    # find parentheses
    par = re.findall(r"\((.*?)\)", note)
    for p in par:
        parts = re.split(r",|and|&|\swith\s|:|;", p)
        for s in parts:
            w = s.strip().strip("." )
            if w:
                foods.append(w)
    # also handle patterns like 'sushi roll: salmon, avocado'
    colon = re.findall(r"(?:[:\-])\s*([^,;\n]+(?:,\s*[^,;\n]+)*)", note)
    # colon matches many; keep only likely food lists
    # skip first if it's the leading time
    return list(dict.fromkeys(foods))


def infer_timestamps(note):
    # find explicit HH:MMUTC times
    times = re.findall(r"(\d{1,2}:\d{2}UTC)", note)
    return times


with open('/srv/ai-server/data/training_data.json', 'r') as f:
    existing = json.load(f)

new_entries = []
for i, note in enumerate(variant_notes):
    foods = extract_foods(note)
    times = infer_timestamps(note)
    main_label = 'food_event'
    # determine label from note keywords
    if 'breakfast' in note.lower() or 'pancake' in note.lower() or 'omelette' in note.lower() or 'bagel' in note.lower():
        main_label = 'breakfast'
    elif 'lunch' in note.lower() or 'sandwich' in note.lower() or 'sushi' in note.lower():
        main_label = 'lunch'
    elif 'dinner' in note.lower() or 'roast' in note.lower() or 'stew' in note.lower() or 'curry' in note.lower():
        main_label = 'dinner'
    elif 'snack' in note.lower() or 'snacked' in note.lower() or 'late snack' in note.lower():
        main_label = 'snack'
    elif 'drink' in note.lower() or 'drank' in note.lower() or 'coffee' in note.lower() or 'tea' in note.lower():
        main_label = 'drink'

    # base entry
    entry = {"note": note, "output": []}
    if times:
        # primary event at first time
        entry["output"].append({"timestamp": times[0], "label": main_label, "food": foods} if foods else {"timestamp": times[0], "label": main_label})
    else:
        # no explicit time: use placeholder None (model should handle)
        entry["output"].append({"label": main_label, "food": foods} if foods else {"label": main_label})

    new_entries.append(entry)

    # create 2 variations: shift time by +/-5 minutes and add/remove one food
    for var in [ -5, 7]:
        note_time_matches = re.findall(r"(\d{1,2}:\d{2})UTC", note)
        if note_time_matches:
            hhmm = note_time_matches[0]
            t = datetime.strptime(hhmm, "%H:%M")
            tvar = (t + timedelta(minutes=var)).time()
            tvar_str = tvar.strftime("%H:%MUTC")
            vnote = re.sub(r"\d{1,2}:\d{2}UTC", tvar_str, note, count=1)
            vfoods = list(foods)
            # add or remove a food token for variety
            if var == -5:
                # remove last if exists
                if vfoods:
                    vfoods = vfoods[:-1]
            else:
                # add a generic item
                vfoods = vfoods + ["salt"] if vfoods else ["water"]
            ventry = {"note": vnote, "output": []}
            ventry["output"].append({"timestamp": tvar_str, "label": main_label, "food": vfoods} if vfoods else {"timestamp": tvar_str, "label": main_label})
            new_entries.append(ventry)
        else:
            # no time; create a slight text variation (add drank water)
            vnote = note + " Drank water afterwards."
            ventry = {"note": vnote, "output": []}
            ventry["output"].append({"label": main_label, "food": (foods + ["water"]) if foods else ["water"]})
            new_entries.append(ventry)

# append to existing
combined = existing + new_entries
with open('/srv/ai-server/data/training_data.json', 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Appended {len(new_entries)} entries. Total now: {len(combined)}")
