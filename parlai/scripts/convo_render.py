import os
import sys
import json
import random
import argparse
import tempfile

# Constant Denoting the End of a Conversation
END_OF_CONVO = "EOC"


def gen_ul_style():
    """ Generate general ul tag style"""
    return (
        "\t\tul{\n\t\t\tlist-style: none;\n\t\t\tmargin: 0;\n\t\t\tpadding: 0;\n\t\t}\n"
    )


def gen_ul_li_style():
    """ Generate style for the li items for the conversations"""
    return "\t\tul li{\n\t\t\tdisplay:inline-block;\n\t\t\tclear: both;\n\t\t\tpadding: 20px;\n\t\t\tborder-radius: 30px;\n\t\t\tmargin-bottom: 2px;\n\t\t\tfont-family: Helvetica, Arial, sans-serif;\n\t\t}\n"


def gen_breaker_style():
    """ Generate style for the break li item between conversations"""
    return "\t\t.breaker{\n\t\t\tcolor: #bec3c9;\n\t\t\tdisplay: block;\n\t\t\theight: 20px;\n\t\t\tmargin: 20px 20px 20px 20px;\n\t\t\ttext-align: center;\n\t\t\ttext-transform: uppercase;\n\t\t}\n"


def gen_speaker_styles(other_speaker, human_speaker):
    """
    Generate style string for the speakers
    :param other_speaker: The title of the model (grey boxes)
    :param human_speaker: Human speaker in the dialogs (blue boxes)

    :return: Style string for the speakers
    """
    other_style = (
        "\t\t."
        + other_speaker
        + "{\n\t\t\tbackground: #eee;\n\t\t\tfloat: left;\n\t\t}\n"
    )
    human_style = (
        "\t\t."
        + human_speaker
        + "{\n\t\t\tbackground: #0084ff;\n\t\t\tcolor: #fff;\n\t\t\tfloat: right;\n\t\t}\n"
    )

    return other_style + human_style


def gen_style_str(height, width, other_speaker, human_speaker):
    """
    Generate the style section string of the HTML
    :param height: Height of the HTML page
    :param width: Width of the HTML page
    :param other_speaker: The title of the model (grey boxes)
    :param human_speaker: Human speaker in the dialogs (blue boxes)

    :return: The <style> section string
    """
    ul_style = gen_ul_style()
    ul_li_style = gen_ul_li_style()
    speaker_styles = gen_speaker_styles(other_speaker, human_speaker)
    breaker_style = gen_breaker_style()
    style_str = (
        '\t<style type="text/css">\n\t\t@media print {\n\t\t\t@page { margin: 0; size:'
        + str(width)
        + "in "
        + str(height)
        + "in;}\n\t\t}\n"
        + ul_style
        + ul_li_style
        + speaker_styles
        + breaker_style
        + "\n\t</style>\n"
    )

    return style_str


def gen_convo_ul(conversations):
    """
    Generate the ul section of the HTML for the conversations
    :param conversation: The conversation to be rendered (after pre-processing)

    :return: The string generating the list in HTML
    """
    ul_str = "\t<ul>\n"
    for speaker, speech in conversations:
        if speaker == END_OF_CONVO:
            ul_str += '\t\t<li class=\"breaker\">' + "End of Conversation" + "</li>\n"
        else:
            ul_str += "\t\t<li class=\"" + speaker + "\">" + speech + "</li>\n"
    ul_str += "\t</ul>\n"

    return ul_str


def gen_html(conversations, height, width, title, other_speaker, human_speaker):
    """
    Generate HTML string for the given conversation
    :param conversation: The conversation to be rendered (after pre-processing)
    :param height: Height of the HTML page
    :param width: Width of the HTML page
    :param title: Title of the HTML page
    :param other_speaker: The title of the model (grey boxes)
    :param human_speaker: Human speaker in the dialogs (blue boxes)

    :return: HTML string for the desired conversation
    """
    html_str = (
        '<html>\n<head>\n\t<meta http-equiv="content-type" content="text/html; charset=utf-8">\n\t<title>'
        + title
        + "</title>\n"
    )
    style_str = gen_style_str(height, width, other_speaker, human_speaker)
    html_str += style_str + "</head>\n"
    body_str = "<body>\n" + gen_convo_ul(conversations) + "</body>"
    html_str += body_str + "\n</html>"
    return html_str


def pre_process(fname, num_ex, alt_speaker):
    """
    Pre-process the given file to bring the conversation in a certain format
    :param fname: File name to be processed
    :param num_ex: Number of conversations to be used
    :param alt_speaker: Name of other speaker to be used

    :return: List of tuples of the form: (speaker, speech)
    """
    conversation = []
    with open(fname) as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:num_ex]
        for line in lines:
            data = json.loads(line)
            dialogue = data["dialog"]
            for item in dialogue:
                if item["speaker"] == "human_evaluator":
                    speaker = "human"
                else:
                    speaker = alt_speaker
                conversation += [(speaker, item["text"])]
            conversation += [(END_OF_CONVO, END_OF_CONVO)]

    return conversation


def prBlueBG(text):
    """
    Print given in text with a blue background
    :param text: The text to be printed
    """
    print("\033[44m{}\033[0m".format(text), sep="")


def display_cli(conversations, alt_speaker, human_speaker):
    """
    Display the conversations on the Command Line
    :param conversations: The dialogs to be displayed
    :param alt_speaker: Name of other speaker to be used
    :param human_speaker: Name of human speaker to be used
    """
    for speaker, speech in conversations:
        if speaker == END_OF_CONVO:
            print("-" * 20 + "END OF CONVERSATION" + "-" * 20)
        elif speaker == alt_speaker:
            print("%-15s: %s" % (speaker[:15], speech))
        else:
            prBlueBG("%-15s: %s" % (speaker[:15], speech))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Conversation Rendering arguments"
    )
    parser.add_argument("i", help="Input file to read conversations from")
    parser.add_argument(
        "-o",
        help="Output file to write conversations to. One of [.pdf, .png, .html] only",
    )
    parser.add_argument("-wd", help="Width of output file", type=int, default=8)
    parser.add_argument("-ht", help="Height of output file", type=int, default=9.5)
    parser.add_argument(
        "-ne", help="Number of conversations to render", type=int, default=10
    )

    args = parser.parse_args()
    input_file, output_file = args.i, args.o
    height, width = args.ht, args.wd
    alt_speaker = input_file.split('/')[-1][:-6]

    dialogs = pre_process(input_file, args.ne, alt_speaker)

    if output_file is None:
        # CLI
        display_cli(dialogs, alt_speaker, "human")
    else:
        extension = output_file.split(".")[-1]
        if extension not in ["html", "pdf", "png"]:
            raise Exception(
                "Extension not specified/supported. Specify one of '.html', '.pdf' or '.png' output files"
            )
        html_str = gen_html(
            dialogs, height, width, "Rendered HTML", alt_speaker, "human"
        )
        if extension == "html":
            # save to output
            file_handle = open(output_file, "w")
            file_handle.write(html_str)
            file_handle.close()
        else:
            # create temp dir
            with tempfile.TemporaryDirectory() as tmpdir:
                fname = tmpdir + "/interim.html"  # save html to interim.html in tmpdir
                file_handle = open(fname, "w")
                file_handle.write(html_str)
                if extension == "pdf":
                    cmd = (
                        '/Applications/Google\\ Chrome.app/Contents/MacOS//Google\\ Chrome --headless --crash-dumps-dir=/tmp --print-to-pdf="'
                        + output_file
                        + '" '
                        + fname
                    )
                else:
                    cmd = (
                        '/Applications/Google\\ Chrome.app/Contents/MacOS//Google\\ Chrome --headless --crash-dumps-dir=/tmp --screenshot="'
                        + output_file
                        + '" '
                        + fname
                    )
                os.system(cmd)
                file_handle.close()
