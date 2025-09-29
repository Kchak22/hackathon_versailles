# AGENT 1 == NL convertor
**desc :** agent qui transforme intention utilisateur en json exploitable par système

**input :** une réponse utilisateur simple ou un json formatter (iso output) ET une réponse utilisateur

**output :** 
                {
                "language": null,
                "party": {
                    "adults_count": null,
                    "children_count": 0,
                    "children_ages": []
                },
                "mobility": {
                    "needs": "none",
                    "max_walking_minutes": null
                },
                "accessibility": {
                    "notes": ""
                },
                "visit_date": null,
                "time_window": {
                    "start": null,
                    "end": null
                },
                "duration_hours": null,
                "visiting": {
                    "city": "",
                    "country": ""
                },
                "visitor_familiarity": "first_time",
                "pace": "balanced",
                "break_frequency_minutes": 0,
                "interests": {
                    "palace_state_apartments": 0,
                    "palace_hall_of_mirrors": 0,
                    "palace_kings_private": 0,
                    "gardens_general": 0,
                    "musical_fountains_show": 0,
                    "trianon": 0,
                    "queens_hamlet": 0,
                    "temporary_exhibitions": 0,
                    "family_kids_activities": 0,
                    "photography": 0,
                    "history_depth": 0,
                    "shopping_souvenirs": 0,
                    "food_experience": 0,
                    "mythology_references": 0
                },
                "constraints": {
                    "crowd_tolerance": "medium",
                    "weather_tolerance": "sun_ok",
                    "quiet_preference": "medium",
                    "stairs_tolerance": "medium",
                    "must_see": [],
                    "avoid": [],
                    "photo_priority": "medium"
                },
                "meals_services": {
                    "meal_plan": "none",
                    "dietary": "none",
                    "coffee_break_preference": "none"
                },
                "tickets_logistics": {
                    "ticket_status": "none",
                    "pass_type": "none",
                    "skip_the_line_priority": "medium",
                    "bag_luggage": "none",
                    "audio_guide": "maybe",
                    "guided_tour_interest": "none"
                },
                "meta": {
                    "notify_about_events": false,
                    "share_location_day_of_visit": false,
                    "notes": ""
                },
                "missing_fields": [],
                "isMissingField": false
                }




# AGENT 2 == Agent question
**desc :** agent qui va demander des précisions à l'utilisateurs sur les élemnts obligatoires du json, ainsi que les préférences utilisateurs
en anguage naturelle dans la langue de l'utilisateur

**input :** json formater 
        {'language': 'fr', 'fields_to_ask': ['party.adults_count', 'visit_date', 'time_window.end'], 'interests': {'palace_state_apartments': 0, 'palace_hall_of_mirrors': 0, 'palace_kings_private': 0, 'gardens_general': 0, 'musical_fountains_show': 0, 'trianon': 0, 'queens_hamlet': 0, 'temporary_exhibitions': 0, 'family_kids_activities': 0, 'photography': 0, 'history_depth': 0, 'shopping_souvenirs': 0, 'food_experience': 0, 'mythology_references': 0}, 'constraints': {'crowd_tolerance': 'medium', 'weather_tolerance': 'sun_ok', 'quiet_preference': 'medium', 'stairs_tolerance': 'medium', 'must_see': [], 'avoid': [], 'photo_priority': 'medium'}, 'meals_services': {'meal_plan': 'quick_snack', 'dietary': 'none', 'coffee_break_preference': 'none'}, 'tickets_logistics': {'ticket_status': 'none', 'pass_type': 'none', 'skip_the_line_priority': 'medium', 'bag_luggage': 'none', 'audio_guide': 'maybe', 'guided_tour_interest': 'none'}, 'time_window': {'start': '10:00', 'end': None}}
output : une question en language naturelle à poser à l'utilisateur qu'on peut passer à l'agent 1

# AGENT 3 == RAG Agent 
**desc :** agent qui prend en entrée les préférences de l'utilisateur, et sort pour chaque préférence k (par défaut k = 5) objets les plus pertinents ainsi que leur score.

**input de la fonction search_objects :** liste des préférences de l'utilisateur.

**output :** Un dictionnaire dont les clées sont les préférences données en output et les valeurs sont des listes de dictionnaires dont chaque dictionnaire contient la hiérarchie pour trouver l'objet (group - subgroup - place - object) ainsi que le score de pertinence associé.

# Pipeline
user prompt -> agent 1 -> json formatter -> agent 2 -> réponse user + json formatter -> Agent  -> Json final
