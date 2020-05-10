from parlai.utils.strings import colorize, name_to_classname
import importlib
import os
import sys


def display_image():
    if os.environ.get('PARLAI_DISPLAY_LOGO') == 'OFF':
        return
    logo = colorize('ParlAI - Dialogue Research Platform', 'labels')
    print(logo)


def Parlai():
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        print("no command given")
        exit()

    map = {'train': 'train_model', 'eval': 'eval_model'}
    if command in map:
        command = map[command]

    try:
        module_name = "parlai.scripts.%s" % command
        module = importlib.import_module(module_name)
        class_name = name_to_classname(command)
        model_class = getattr(module, class_name)
    except ImportError:
        print(command + " not found")
        exit()

    display_image()
    sys.argv = sys.argv[1:]  # remove parlai arg
    model_class.main()


if __name__ == '__main__':
    Parlai()
