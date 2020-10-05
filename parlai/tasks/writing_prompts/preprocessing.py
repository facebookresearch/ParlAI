import os
from collections import OrderedDict
from random import random, shuffle, sample

import fire
import more_itertools
from blingfire import text_to_sentences
from jsonlines import jsonlines


def replace_characters(prompt):
    """ Replace special characters.
    """
    prompt = prompt.replace("<newline>", " ").replace("\n", " ").replace("\t", " ").replace("|", " ")
    return prompt

def ensure_dir(file_path):
    """ Make sure the output path exists.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class WritingPrompts(object):
    """Convert WritingPrompts prompts and text files into conversation json format for ParlAI. """

    def parlai(self, prompts_file: str, stories_file: str, output_file: str, label_candidates: int = 0):

        print(f"Write output to: {output_file}")
        ensure_dir(output_file)

        print(f"Read prompts from: {prompts_file}")
        print(f"Read stories from: {stories_file}")

        episodes_list = []
        sentence_list = []

        with open(prompts_file, encoding="utf8") as prompts, open(stories_file, encoding="utf8") as stories:
            for i, (prompt, story) in enumerate(zip(prompts, stories)):

                # print(f"Prompt and Story: {prompt} - {story}")

                dialog_list = []

                prompt = replace_characters(prompt)

                story = replace_characters(story)

                sentences = text_to_sentences(story).split('\n')
                # print(f"Sentences: {sentences}")

                sentences = [prompt] + sentences

                for chunk in more_itertools.chunked(sentences, n=2):

                    if len(chunk) == 2:
                        text = chunk[0]
                        labels = chunk[1]
                        # print(f"Text: {text}, Labels: {labels}")

                        dialog_list.append({"text": text, "labels": labels})
                        sentence_list.append(text)
                        sentence_list.append(labels)

                episodes_list.append(dialog_list)

        if label_candidates > 0:
            for episode in episodes_list:
                for dialog_line in episode:
                    labels_candidates_list = [dialog_line["labels"]] + list(
                        sample(sentence_list, label_candidates))
                    shuffle(labels_candidates_list)
                    dialog_line["label_candidates"] = labels_candidates_list

        with open(output_file, 'w', encoding="utf8", newline='\n') as out_file:
            for episode in episodes_list:
                for i, dialog_line in enumerate(episode, start=1):
                    if "label_candidates" in dialog_line:
                        if i < len(episode):
                            line_to_write = f"text:{dialog_line['text']}\tlabels:{dialog_line['labels']}\t" \
                                            f"label_candidates:{'|'.join(dialog_line['label_candidates'])}\n"
                        else:
                            line_to_write = f"text:{dialog_line['text']}\tlabels:{dialog_line['labels']}\t" \
                                            f"label_candidates:{'|'.join(dialog_line['label_candidates'])}" \
                                            f"\tepisode_done:True\n"
                    else:
                        if i < len(episode):
                            line_to_write = f"text:{dialog_line['text']}\tlabels:{dialog_line['labels']}\n"
                        else:
                            line_to_write = f"text:{dialog_line['text']}\tlabels:{dialog_line['labels']}" \
                                            f"\tepisode_done:True\n"

                    out_file.write(line_to_write)

    def json(self, prompts_file: str, stories_file: str, output_file: str):

        print(f"Write output to: {output_file}")
        ensure_dir(output_file)

        print(f"Read prompts from: {prompts_file}")
        print(f"Read stories from: {stories_file}")

        stories_list = []

        with open(prompts_file, encoding="utf8") as prompts, open(stories_file, encoding="utf8") as stories:
            for i, (prompt, story) in enumerate(zip(prompts, stories)):

                prompt = prompt.replace("<newline>","\n")
                story = story.replace("<newline>", "\n")

                story_dict = OrderedDict()
                story_dict["id"] = i
                story_dict["title"] = prompt
                story_dict["text"] = story

                stories_list.append(story_dict)

        with jsonlines.open(output_file, mode='w') as writer:

            for story in stories_list:
                writer.write(story)


if __name__ == '__main__':
    fire.Fire(WritingPrompts)
