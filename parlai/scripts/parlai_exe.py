from parlai.utils.strings import colorize, name_to_classname
import importlib
import sys

def display_image():
    if os.environ.get('PARLAI_DISPLAY_LOGO') = 'OFF':
        return

    image1 = (
        "         \\              \n" +
        " \\      (o>             \n" +
        " (o>     //\   ")
    image2 = "ParlAI    \n"
    image3 = ("_(()_____V_/_________    \n" +
        " ||      ||              \n" + 
        "         ||              \n")
    image1 = image1.replace('\\', "\\\\")
    image1 = image1.replace('/\\\\', "/\\")
    image = (colorize(image1, 'text')
             + colorize(image2, 'bold_text')
             + colorize(image3, 'text'))
    print(image)
          
def Parlai():
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        print("no command given")
        exit()
        
    try:
        module_name = "parlai.scripts.%s" % command
        module = importlib.import_module(module_name)
        class_name = name_to_classname(sys.argv[1])
        model_class = getattr(module, class_name)
    except ImportError:
        print(command + " not found")
        exit()

    display_image()
    sys.argv = sys.argv[1:]     # remove parlai arg
    model_class.main()

if __name__ == '__main__':
    Parlai()
