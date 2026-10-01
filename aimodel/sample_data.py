"""A starter training set for the writer.

Your real training data comes from your own chats (`/export`). Until you have
enough of those, this builds a clean, general dataset in exactly the same format
so you can train a first writer and see the whole pipeline work:

    python -m aimodel.sample_data --out sample_data

It teaches a model simple facts (animals, countries, science, how-tos, Python,
a made-up person's notes), then lets the model answer practice questions in
many forms. Every example is: question, the evidence sentences, the answer
written from that evidence. A writer trained on it learns the *skill* of
wording an answer from evidence; the facts themselves are only examples. A
few `your correction` examples show how your own /bad corrections look.
"""

from __future__ import annotations

import argparse
import os

from .model import LearningModel
from .writer import export_training_data

# name, class, where it lives, what it eats
ANIMALS = [
    ("red fox", "mammal", "forests", "small animals and berries"),
    ("gray wolf", "mammal", "forests", "meat"),
    ("cow", "mammal", "farms", "grass"),
    ("horse", "mammal", "grasslands", "grass"),
    ("sheep", "mammal", "grasslands", "grass"),
    ("goat", "mammal", "mountains", "plants"),
    ("pig", "mammal", "farms", "grain and vegetables"),
    ("rabbit", "mammal", "grasslands", "grass and leaves"),
    ("elephant", "mammal", "Africa and Asia", "plants"),
    ("giraffe", "mammal", "African savannas", "leaves"),
    ("lion", "mammal", "African grasslands", "meat"),
    ("tiger", "mammal", "Asian forests", "meat"),
    ("zebra", "mammal", "African grasslands", "grass"),
    ("kangaroo", "mammal", "Australia", "grass and leaves"),
    ("koala", "mammal", "Australia", "eucalyptus leaves"),
    ("giant panda", "mammal", "mountain forests in China", "bamboo"),
    ("polar bear", "mammal", "the Arctic", "seals"),
    ("brown bear", "mammal", "forests", "fish and berries"),
    ("dolphin", "mammal", "the ocean", "fish"),
    ("blue whale", "mammal", "the ocean", "krill"),
    ("camel", "mammal", "deserts", "desert plants"),
    ("squirrel", "mammal", "forests", "nuts and seeds"),
    ("hedgehog", "mammal", "gardens and forests", "insects"),
    ("fruit bat", "mammal", "caves and forests", "fruit"),
    ("bald eagle", "bird", "North America", "fish and small animals"),
    ("sparrow", "bird", "towns and fields", "seeds and insects"),
    ("owl", "bird", "forests", "mice and insects"),
    ("penguin", "bird", "Antarctica", "fish"),
    ("parrot", "bird", "tropical forests", "fruit and seeds"),
    ("duck", "bird", "ponds and rivers", "water plants and insects"),
    ("swan", "bird", "lakes and rivers", "water plants"),
    ("hummingbird", "bird", "the Americas", "nectar"),
    ("ostrich", "bird", "African grasslands", "plants and insects"),
    ("crow", "bird", "fields and towns", "seeds, insects and small animals"),
    ("crocodile", "reptile", "rivers and swamps", "fish and other animals"),
    ("sea turtle", "reptile", "the ocean", "jellyfish and seagrass"),
    ("cobra", "reptile", "Asia and Africa", "small animals"),
    ("chameleon", "reptile", "forests", "insects"),
    ("green iguana", "reptile", "tropical forests", "leaves and fruit"),
    ("komodo dragon", "reptile", "Indonesia", "deer and other animals"),
    ("salmon", "fish", "rivers and the ocean", "insects and small fish"),
    ("great white shark", "fish", "the ocean", "fish and seals"),
    ("clownfish", "fish", "coral reefs", "algae and tiny animals"),
    ("goldfish", "fish", "ponds", "plants and insects"),
    ("tuna", "fish", "the ocean", "smaller fish"),
    ("honeybee", "insect", "gardens and meadows", "nectar and pollen"),
    ("butterfly", "insect", "gardens and meadows", "nectar"),
    ("ladybug", "insect", "gardens", "aphids"),
    ("dragonfly", "insect", "ponds and rivers", "small insects"),
    ("grasshopper", "insect", "grasslands", "grass"),
    ("frog", "amphibian", "ponds and wetlands", "insects"),
    ("salamander", "amphibian", "damp forests", "insects and worms"),
    ("spider", "arachnid", "gardens and forests", "insects"),
]

KINDS = (
    "Mammals are warm-blooded animals. Birds are animals that have feathers. "
    "Reptiles are cold-blooded animals. Fish are animals that live in water. "
    "Insects are small animals with six legs. Amphibians are animals that live on land and in water. "
    "Arachnids are animals with eight legs. Animals are living things. Plants are living things. "
    "Whales are not fish. Dolphins are not fish. Bats are not birds. Snakes are not mammals. "
    "Spiders are not insects. Penguins are not mammals."
)

# country, capital, continent
COUNTRIES = [
    ("France", "Paris", "Europe"), ("Germany", "Berlin", "Europe"), ("Italy", "Rome", "Europe"),
    ("Spain", "Madrid", "Europe"), ("Portugal", "Lisbon", "Europe"), ("Greece", "Athens", "Europe"),
    ("Japan", "Tokyo", "Asia"), ("India", "New Delhi", "Asia"), ("China", "Beijing", "Asia"),
    ("Thailand", "Bangkok", "Asia"), ("South Korea", "Seoul", "Asia"), ("Indonesia", "Jakarta", "Asia"),
    ("Egypt", "Cairo", "Africa"), ("Kenya", "Nairobi", "Africa"), ("Nigeria", "Abuja", "Africa"),
    ("Morocco", "Rabat", "Africa"), ("Brazil", "Brasilia", "South America"),
    ("Argentina", "Buenos Aires", "South America"), ("Peru", "Lima", "South America"),
    ("Canada", "Ottawa", "North America"), ("Mexico", "Mexico City", "North America"),
    ("Cuba", "Havana", "North America"), ("Australia", "Canberra", "Oceania"),
    ("New Zealand", "Wellington", "Oceania"),
]

# a cause and the "why" question that asks for it
CAUSES = [
    ("The sky is blue because air scatters blue light more than red light.", "Why is the sky blue?"),
    ("Ice floats on water because ice is less dense than liquid water.", "Why does ice float on water?"),
    ("Plants need sunlight because they use light energy to make food.", "Why do plants need sunlight?"),
    ("Leaves look green because chlorophyll reflects green light.", "Why are leaves green?"),
    ("Thunder follows lightning because light travels faster than sound.",
     "Why does thunder follow lightning?"),
    ("The seasons change because the Earth is tilted on its axis.", "Why do the seasons change?"),
    ("The moon has phases because we see different parts of its sunlit side as it orbits the Earth.",
     "Why does the moon have phases?"),
    ("Birds can fly because they have wings and light bones.", "Why can birds fly?"),
    ("Bread rises because yeast makes gas that fills the dough with bubbles.", "Why does bread rise?"),
    ("Metal feels cold because it moves heat away from your hand quickly.",
     "Why does metal feel cold?"),
]

# steps and the "how" question that asks for them
STEPS = [
    ("To make tea, boil water in a kettle. Then put a tea bag in a cup. Pour the hot water over the "
     "tea bag. Finally, wait three minutes and remove the bag.", "How do I make tea?"),
    ("To plant a seed, dig a small hole. Then place the seed inside. After that, cover it with soil. "
     "Finally, water it gently.", "How do I plant a seed?"),
    ("To wash your hands, wet them with clean water. Then apply soap. After that, rub your hands "
     "together for twenty seconds. Finally, rinse and dry them.", "How do I wash my hands?"),
    ("To boil an egg, place it in a pot of water. Then heat the water until it boils. After that, "
     "cook the egg for ten minutes. Finally, cool it in cold water.", "How do I boil an egg?"),
    ("To send an email, open your mail app. Then write a new message. After that, add the address and "
     "a subject. Finally, press send.", "How do I send an email?"),
]

FACTS = (
    "Photosynthesis is the process plants use to make food from light, water and carbon dioxide. "
    "Gravity is the force that pulls objects toward the Earth. The Sun is a star. The Earth is a planet. "
    "Mars is the fourth planet from the Sun. Jupiter is the largest planet in the solar system. "
    "The Moon orbits the Earth. Water is a liquid. Ice is frozen water. "
    "Python is a programming language. Python is easy to read. A list is an ordered collection of items. "
    "Lists are mutable. Tuples are immutable. A dictionary stores values by key. "
    "A function is a reusable block of code. A variable is a name that holds a value."
)
FACT_QUESTIONS = [
    "What is photosynthesis?", "What is gravity?", "What is the Sun?", "What is the Earth?",
    "What is Mars?", "What is Jupiter?", "Tell me about the Moon", "What is ice?",
    "What is Python?", "What is a list?", "Are lists mutable?", "What are tuples?",
    "What is a dictionary?", "What is a function?", "What is a variable?",
]

# A made-up person, so the writer practises talking about "you": notes become "your ...".
PERSONA = (
    "My dog is called Rex. Rex is a golden retriever who loves swimming. My sister lives in Lisbon. "
    "I like tea. I like building small things. My favourite programming language is Python because "
    "it is easy to read. My brother lives in Toronto. I like long walks."
)
PERSONA_QUESTIONS = [
    "What is my dog called?", "Tell me about my dog", "What is Rex?", "Where does my sister live?",
    "What do I like?", "What is my favourite programming language?", "Where does my brother live?",
]

# What /bad corrections look like: the question, the evidence, the answer you typed.
CORRECTIONS = [
    ("What is my dog called?", ["Your dog is called Rex."], "Your dog is called Rex."),
    ("Where does my sister live?", ["Your sister lives in Lisbon."], "Your sister lives in Lisbon."),
    ("What is Rex?", ["Rex is a golden retriever who loves swimming.", "Your dog is called Rex."],
     "Rex is your dog, a golden retriever who loves swimming."),
    ("Why is the sky blue?", ["The sky is blue because air scatters blue light more than red light."],
     "Air scatters blue light more than red light, so the sky is blue."),
    ("What is the capital of France?", ["Paris is the capital of France."], "Paris is the capital of France."),
    ("What does the panda eat?", ["The giant panda eats bamboo."], "The giant panda eats bamboo."),
    ("Where does the kangaroo live?", ["The kangaroo lives in Australia."],
     "The kangaroo lives in Australia."),
    ("Is a cow an animal?", ["The cow is a mammal.", "Mammals are warm-blooded animals.",
                              "Animals are living things."],
     "Yes. A cow is a mammal, and mammals are animals."),
    ("What is a list?", ["A list is an ordered collection of items.", "Lists are mutable."],
     "A list is an ordered collection of items, and lists are mutable."),
    ("Why does ice float on water?", ["Ice floats on water because ice is less dense than liquid water."],
     "Ice floats on water because ice is less dense than liquid water."),
    ("How do I make tea?", ["To make tea, boil water in a kettle.", "Then put a tea bag in a cup.",
                             "Pour the hot water over the tea bag.", "Finally, wait three minutes and remove the bag."],
     "Boil water in a kettle, put a tea bag in a cup and pour the hot water over it, then wait three "
     "minutes and remove the bag."),
    ("What do I like?", ["You like tea.", "You like building small things."],
     "You like tea and building small things."),
]


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def _animal_text(name: str, cls: str, habitat: str, diet: str) -> str:
    return (f"The {name} is {_article(cls)} {cls}. The {name} lives in {habitat}. "
            f"The {name} eats {diet}.")


def animal_questions() -> list[str]:
    """Many ways of asking about each animal."""
    out = []
    for name, cls, _habitat, _diet in ANIMALS:
        out += [f"What is the {name}?", f"What is {_article(name)} {name}?", f"Tell me about the {name}",
                f"Where does the {name} live?", f"What does the {name} eat?",
                f"Is the {name} {_article(cls)} {cls}?", f"Is {_article(name)} {name} an animal?"]
    return out


def country_questions() -> list[str]:
    out = []
    for country, capital, _continent in COUNTRIES:
        out += [f"What is the capital of {country}?", f"Where is {country}?", f"What is {capital}?"]
    return out


def build_sample_model(seed: int = 0) -> tuple[LearningModel, list[str]]:
    """A model that has read the starter facts, and the practice questions to ask it."""
    m = LearningModel(seed=seed)
    m.add_document(" ".join(_animal_text(*a) for a in ANIMALS), "animals.txt")
    m.add_document(KINDS, "animal_kinds.txt")
    m.add_document(" ".join(f"{cap} is the capital of {c}. {c} is a country in {cont}."
                            for c, cap, cont in COUNTRIES), "countries.txt")
    m.add_document(" ".join(c for c, _ in CAUSES), "science.txt")
    m.add_document("\n".join(t for t, _ in STEPS), "howto.txt")
    m.add_document(FACTS, "facts.txt")
    m.add_document(PERSONA, "sample_person.txt")
    m.answer_log = [{"q": q, "evidence": ev, "template": ans, "final": ans, "by": "templates",
                     "rating": "bad", "correction": ans} for q, ev, ans in CORRECTIONS]
    questions = (animal_questions() + country_questions() + [q for _, q in CAUSES]
                 + [q for _, q in STEPS] + FACT_QUESTIONS + PERSONA_QUESTIONS)
    return m, questions


ABOUT = """Starter training data for the Aimodel writer (a tiny transformer).

writer_data.jsonl  one example per line: question, evidence (the sentences the answer
                   is built from), answer, weight and kind. `kind` is practice (questions
                   asked of the facts), your correction (weight 3, how your own /bad
                   corrections look) or taught. This is the same format /export writes.
corpus.txt         every sentence the model read, for language practice.

The facts are generic samples (animals, countries, science, how-tos, Python, a made-up
person), not your data. A writer trained on this learns to word answers from evidence.
Train with:  python -m aimodel.train_writer --data sample_data --out writer.npz
Rebuild with: python -m aimodel.sample_data --out sample_data
"""


def export_sample(folder: str, seed: int = 0) -> dict:
    model, questions = build_sample_model(seed)
    info = export_training_data(model, folder, extra_questions=questions, exam_questions=False)
    with open(os.path.join(folder, "ABOUT.txt"), "w", encoding="utf-8") as f:
        f.write(ABOUT)
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write the starter training data for the writer.")
    ap.add_argument("--out", default="sample_data")
    info = export_sample(ap.parse_args(argv).out)
    print(f"Wrote {info['examples']} examples and {info['corpus lines']} lines of text to {info['folder']}/")
    print("By kind:", ", ".join(f"{n} {k}" for k, n in info["by kind"].items()))


if __name__ == "__main__":
    main()
