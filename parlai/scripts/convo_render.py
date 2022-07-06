#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import tempfile
import subprocess
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.io import PathManager

# Constants
END_OF_CONVO = "EOC"
CHROME_PATH = r'/Applications/Google\ Chrome.app/Contents/MacOS//Google\ Chrome'

ALT_EMOJI_IMG = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/facebook/230/parrot_1f99c.png"
HUMAN_EMOJI_IMG = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/apple/76/woman_1f469.png"


def gen_convo_ul(conversations):
    """
    Generate the ul section of the HTML for the conversations.

    :param conversation: The conversation to be rendered (after pre-processing)

    :return: The string generating the list in HTML
    """
    ul_str = f"\t<ul>\n"
    for speaker, speech in conversations:
        if speaker == END_OF_CONVO:
            ul_str += f"\n\t  <li class=\"breaker\"><hr/></li>\n"
        else:
            ul_str += f"""
    <li>
        <div class="{speaker}_img_div">
            <img class="{speaker}_img">
        </div>
        <div class="{speaker}_p_div">
            <p class="{speaker}">{speech}</p>
        </div>
        <div class="clear"></div>
    </li>
    """
    ul_str += "\t</ul>"

    return ul_str


def gen_html(
    conversations,
    height,
    width,
    title,
    other_speaker,
    human_speaker,
    user_icon,
    alt_icon,
):
    """
    Generate HTML string for the given conversation.

    :param conversation:
        The conversation to be rendered (after pre-processing)
    :param height:
        Height of the HTML page
    :param width:
        Width of the HTML page
    :param title:
        Title of the HTML page
    :param other_speaker:
        The title of the model (grey boxes)
    :param human_speaker:
        Human speaker in the dialogs (blue boxes)

    :return: HTML string for the desired conversation
    """
    html_str = f"""<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <title> {title} </title>
    <style type="text/css">
        @media print{{
            @page{{ margin: 0; size: {str(width)}in {str(height)}in; }}
        }}
        ul{{
          list-style: none;
        }}
        .{other_speaker}_img_div{{
          display: inline-block;
          float: left;
          margin: 18px 5px 0px -25px;
        }}
        .{human_speaker}_img_div{{
          display: inline-block;
          float: right;
          margin: 18px 15px 5px 5px;
        }}
        .{other_speaker}_img{{
            content:url({alt_icon});
        }}
        .{human_speaker}_img{{
            content:url({user_icon});
        }}
        .{other_speaker}_p_div{{
          float: left;
        }}
        .{human_speaker}_p_div{{
          float:right;
        }}
        p{{
          display:inline-block;
          overflow-wrap: break-word;
          border-radius: 30px;
          padding: 10px 10px 10px 10px;
          font-family: Helvetica, Arial, sans-serif;
        }}
        .clear{{
            float: none;
            clear: both;
        }}
        .{other_speaker}{{
                background: #eee;
                float: left;
            }}
        .{human_speaker}{{
            background: #0084ff;
            color: #fff;
            float: right;
        }}
        .breaker{{
            color: #bec3c9;
            display: block;
            height: 20px;
            margin: 20px 20px 20px 20px;
            text-align: center;
            text-transform: uppercase;
        }}
        img{{
          border-radius: 50px;
          width: 50px;
          height: 50px;
        }}
    </style>
</head>
<body>
{gen_convo_ul(conversations)}
</body>
</html>
    """
    return html_str


def pre_process(fname, num_ex, alt_speaker):
    """
    Pre-process the given file to bring the conversation in a certain format.

    :param fname: File name to be processed
    :param num_ex: Number of conversations to be used
    :param alt_speaker: Name of other speaker to be used

    :return: List of tuples of the form: (speaker, speech)
    """
    conversation = []
    with PathManager.open(fname) as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:num_ex]
        for line in lines:
            data = json.loads(line)
            dialogue = data["dialog"]
            for item in dialogue:
                speaker = item[0]['id']
                text = item[0]['text']
                conversation += [(speaker, text)]
            conversation += [(END_OF_CONVO, END_OF_CONVO)]

    return conversation


def prBlueBG(text):
    """
    Print given in text with a blue background.

    :param text: The text to be printed
    """
    print("\033[44m{}\033[0m".format(text), sep="")


def display_cli(conversations, alt_speaker, human_speaker):
    """
    Display the conversations on the Command Line.

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


def setup_args():
    """
    Creates a parser object with some pre-determined arguments.
    """
    parser = ParlaiParser(add_parlai_args=True, description="Render data as HTML")
    conv_render = parser.add_argument_group('Conversation Rendering Arguments')
    conv_render.add_argument(
        "--input", "-i", help="Input file to read conversations from"
    )
    conv_render.add_argument(
        "--output",
        "-o",
        help="Output file to write conversations to. One of [.pdf, .png, .html] only",
    )
    conv_render.add_argument(
        "--width", "-wd", help="Width of output file", type=int, default=8
    )
    conv_render.add_argument(
        "--height", "-ht", help="Height of output file", type=int, default=10
    )
    conv_render.add_argument(
        "--user-icon",
        "-uic",
        help="Absolute Path/URL to user image icon",
        default=HUMAN_EMOJI_IMG,
    )
    conv_render.add_argument(
        "--alt-icon",
        "-aic",
        help="Absolute Path/URL to alternate image icon",
        default=ALT_EMOJI_IMG,
    )
    conv_render.add_argument(
        "--num-examples",
        "-ne",
        help="Number of conversations to render",
        type=int,
        default=10,
    )

    return parser


def check_icon_arg(src, default):
    """
    Checks if icon arguments are valid: either a URL or an absolute path.

    :param src: Source of the icon
    :param default: default value of the icon

    :return: src (possibly pre-pended with "file://")
    """
    if src != default:
        # check if URl
        if not src.startswith('https://') and not src.startswith('http://'):
            # Either a file or incorrect input
            if os.path.isabs(src):
                src = "file://" + src
            else:
                raise IOError(
                    f"Please provide a valid URL or valid *absolute* path to icon: {src}"
                )
    return src


def validate_args(opt):
    """
    Validate the cmdline args passed into the script.

    :param opt: The arguments of te parser

    :return: Returns extension of output file. None if no output file
    """
    if not PathManager.exists(opt['input']):
        raise IOError("Input File does not exist")
    if opt['output'] is None:
        return None
    extension = opt['output'].split(".")[-1]
    if extension not in ["html", "pdf", "png"]:
        raise Exception(
            "Extension not specified/supported. Specify one of '.html', '.pdf' or '.png' output files"
        )
    opt['user_icon'] = check_icon_arg(opt['user_icon'], HUMAN_EMOJI_IMG)
    opt['alt_icon'] = check_icon_arg(opt['alt_icon'], ALT_EMOJI_IMG)
    return extension


def render_convo(opt):
    # Run
    opt.log()
    extension = validate_args(opt)
    input_file, output_file = opt['input'], opt['output']
    height, width = opt['height'], opt['width']
    alt_speaker = input_file.split('/')[-1][:-6]

    dialogs = pre_process(input_file, opt['num_examples'], alt_speaker)

    # Display on CLI
    if output_file is None:
        display_cli(dialogs, alt_speaker, "human")
    # Some form of output file
    else:
        html_str = gen_html(
            dialogs,
            height,
            width,
            "Rendered HTML",
            alt_speaker,
            "human",
            opt['user_icon'],
            opt['alt_icon'],
        )
        if extension == "html":
            # save to output
            with PathManager.open(output_file, "w") as file_handle:
                file_handle.write(html_str)
        else:
            # create temp dir
            with tempfile.TemporaryDirectory() as tmpdir:
                fname = tmpdir + "/interim.html"  # save html to interim.html in tmpdir
                with PathManager.open(fname, "w") as file_handle:
                    file_handle.write(html_str)
                    if extension == "pdf":
                        cmd = (
                            f"{CHROME_PATH} --headless --crash-dumps-dir=/tmp"
                            f"--print-to-pdf=\"{output_file}\" {fname}"
                        )
                    else:
                        cmd = (
                            f"{CHROME_PATH} --headless --hide-scrollbars"
                            f"--crash-dumps-dir=/tmp --window-size"
                            f"={opt['width'] * 100},{opt['height'] * 100}"
                            f"--screenshot=\"{output_file}\" {fname}"
                        )
                    subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
                    )


@register_script('convo_render', hidden=True)
class RenderConversation(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return render_convo(self.opt)


if __name__ == '__main__':
    RenderConversation.main()
