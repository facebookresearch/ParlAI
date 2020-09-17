import os
from random import random, shuffle, sample

import fire
import more_itertools
from blingfire import text_to_sentences


def ensure_dir(file_path):
    """ Make sure
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class WritingPrompts(object):
    """Convert WritingPrompts prompts and text files into conversation json format for ParlAI. """

    def parlaidialo(self, prompts_file: str, stories_file: str, output_file: str, label_candidates: int = 0):

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

                prompt = self.replace_characters(prompt)

                story = self.replace_characters(story)

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
                                            f"label_candidates:{'|'.join(dialog_line['label_candidates'])}"
                        else:
                            line_to_write = f"text:{dialog_line['text']}\tlabels:{dialog_line['labels']}\t" \
                                            f"label_candidates:{'|'.join(dialog_line['label_candidates'])}" \
                                            f"\tepisode_done:True"
                    else:
                        if i < len(episode):
                            line_to_write = f"text:{dialog_line['text']}\tlabels:{dialog_line['labels']}"
                        else:
                            line_to_write = f"text:{dialog_line['text']}\tlabels:{dialog_line['labels']}" \
                                            f"\tepisode_done:True"

                    out_file.write(line_to_write)

    def replace_characters(self, prompt):
        prompt = prompt.replace("<newline>", " ").replace("\t", " ").replace("|", " ")
        return prompt


if __name__ == '__main__':
    fire.Fire(WritingPrompts)
