from collections import defaultdict
from typing import Dict, List, Union, Set

BELIEF_STATE_DELIM = ", "
DOMAINS = [
    "attraction",
    "hotel",
    "hospital",
    "restaurant",
    "police",
    "taxi",
    "train",
]
NAMED_ENTITY_SLOTS = {
    "attraction--name",
    "restaurant--name",
    "hotel--name",
    "bus--departure",
    "bus--destination",
    "taxi--departure",
    "taxi--destination",
    "train--departure",
    "train--destination",
}
NAMED_ENTITY_INTERESTED = {"restaurant--name", "hotel--name", "attraction--name"}


def extract_slot_from_string(slots_string):
    """
    Either ground truth or generated result should be in the format:
    "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
    and this function would reformat the string into list:
    ["dom--slot_type--slot_val", ... ]
    """
    slots_list = []

    if slots_string is None:
        return [], [], [], []

    slot_val_conversion = {
        "centre": "center",
        "3-star": "3",
        "2-star": "2",
        "1-star": "1",
        "0-star": "0",
        "4-star": "4",
        "5-star": "5",
    }

    per_domain_slot_lists = {}
    named_entity_slot_lists = []
    named_entity_slot_interested_lists = []

    # # # remove start and ending token if any
    str_split = slots_string.strip().split()
    if str_split != [] and str_split[0] in ["<bs>", "</bs>"]:
        str_split = str_split[1:]
    if "</bs>" in str_split:
        str_split = str_split[: str_split.index("</bs>")]

    str_split = " ".join(str_split).split(",")
    if str_split[-1] == "":
        str_split = str_split[:-1]
    str_split = [slot.strip() for slot in str_split]

    for slot_ in str_split:
        slot = slot_.split()
        if len(slot) > 2 and slot[0] in DOMAINS:
            domain = slot[0]
            if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                slot_type = slot[1] + " " + slot[2]
                slot_val = " ".join(slot[3:])
            else:
                slot_type = slot[1]
                slot_val = " ".join(slot[2:])
            slot_val = slot_val_conversion.get(slot_val, slot_val)
            # if not slot_val == "dontcare":

            slots_list.append(domain + "--" + slot_type + "--" + slot_val)
            if domain in per_domain_slot_lists:
                per_domain_slot_lists[domain].add(slot_type + "--" + slot_val)
            else:
                per_domain_slot_lists[domain] = {slot_type + "--" + slot_val}
            if domain + "--" + slot_type in NAMED_ENTITY_SLOTS:
                named_entity_slot_lists.append(
                    domain + "--" + slot_type + "--" + slot_val
                )
            if domain + "--" + slot_type in NAMED_ENTITY_INTERESTED:
                named_entity_slot_interested_lists.append(
                    domain + "--" + slot_type + "--" + slot_val
                )

    return (
        slots_list,
        per_domain_slot_lists,
        named_entity_slot_lists,
        named_entity_slot_interested_lists,
    )


def get_dialid2domains(
    messages: List[Dict[str, Union[str, int]]]
) -> Dict[str, Set[str]]:
    """Retrieve a dictionary that maps the dialogue id to the domains that it is part of.


    Args:
        messages (List[Dict[str, Union[str, int]]]): a list of dictionary form of each turn of the MultiWOZ dataset

    Returns:
        Dict[str, Set[str]]: dictionary that maps dialogue id to the domain that it's part of
    """

    dialid2domains = {}
    dialid2turns = defaultdict(list)
    for episode_idx, msg in enumerate(messages):
        dialid2turns[msg['dial_id']].append(msg)

    for dialid, turns in dialid2turns.items():
        # find final turn
        turns = sorted(turns, key=lambda x: x['turn_num'])
        last_turn = turns[-1]

        # extract slots
        slots, _, _, _ = extract_slot_from_string(last_turn['slots_inf'])

        # extract domains
        domains = set([slot.split("--")[0] for slot in slots])

        dialid2domains[dialid] = domains

    return dialid2domains


def my_strip(text):

    while not text[0].isalpha():
        text = text[1:]

    while not text[-1].isalpha():
        text = text[:-1]

    return text
