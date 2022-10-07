import random


def format_context_and_label(context, label, seed=None):
    """
    Transform task to include an instruction with custom labels that are all in natural text
    Seed is for getting back the same templates if needed. Useful for invariant metrics
    """

    templates = [
        (
            f"Conversation: {context} \n Question: What is the dialogue belief state of this conversation?",
            label,
        ),
        (f"Tell me the dialogue belief state of this dialogue: {context}", label),
        (
            f"List out the dialogue belief state of the following conversation in 'domain slot type slot value' format separated by commas: {context}",
            label,
        ),
        (
            f"Here's a conversation: {context} \n Question: What is its dialogue belief state?",
            label,
        ),
        (f"Dialogue: {context} Question: What is its dialogue belief state?", label),
        (
            f"Conversation: {context} Question: What is its dialogue belief state? Don't provide any entities that were not given in the conversation.",
            label,
        ),
    ]

    # what I had before. Is this the cause for the difference between 85 and 78? No.
    # templates = [
    #     (
    #         f"{context} What is the dialogue belief state of this conversation?",
    #         label,
    #     ),
    #     (f"Tell me the dialogue belief state of this dialogue: {context}", label),
    #     (
    #         f"List out the dialogue belief state of the following conversation in 'domain slot type slot value' format separated by commas: {context}",
    #         label,
    #     ),
    #     (
    #         f"Here's a conversation: {context} What is it's dialogue belief state?",
    #         label,
    #     ),
    #     (
    #         f"Here's a conversation: {context} What is it's dialogue belief state? Don't provide any entities that were not given in the conversation.",
    #         label,
    #     ),
    # ]
    if seed:
        random.seed(seed)
    template, label = random.choice(templates)

    return template, label
