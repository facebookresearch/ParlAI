#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class EntityMutator(object):
    SUPPORTED_MODES = {"scramble", "gibberish", "delex", "real"}
    SUPPORTED_ENTITY_SOURCEs = {"internal", "external"}
    ALLOWLIST_GIBBERISH = list("abcdefghijklmnopqrstuvwxyz")

    def __init__(self, opt, rng):
        self.opt = opt
        self.rng = rng

    def scramble(self, entity_name):
        modified_name = list(entity_name)
        self.rng.shuffle(modified_name)
        modified_name = "".join(modified_name)
        return modified_name

    def shuffle_words(self, entity_name):
        sub_words = entity_name.split(" ")
        sub_words = sub_words * 5
        self.rng.shuffle(sub_words)
        num_words = self.rng.randint(3, 10)
        return " ".join(sub_words[:num_words])

    def create_madeup_entities(self, entities):
        sub_words = []
        for entity in self.rng.choice(entities, 10):
            sub_words += entity.split(" ")
        self.rng.shuffle(sub_words)
        num_words = self.rng.randint(4, 15)
        return " ".join(sub_words[:num_words])

    def create_gibberish_entity(self, entity_name):
        min_num_words = 3
        max_num_words = 12

        min_word_len = 3
        max_word_len = 20
        num_entities = self.rng.randint(min_num_words, max_num_words)
        gibberish_entities = []
        for _ in range(num_entities):
            str_len = self.rng.randint(min_word_len, max_word_len)
            gibberish_entities.append(
                "".join(self.rng.choice(self.ALLOWLIST_GIBBERISH, str_len))
            )

        return self.rng.choice(gibberish_entities)
        # return " ".join(gibberish_entities)

    def delex(self, tag="a", basename="name"):
        return f"{basename}-{tag}"
