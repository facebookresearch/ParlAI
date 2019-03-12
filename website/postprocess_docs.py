import os

if __name__ == '__main__':
    try:
        os.rename("html", "docs")
    except FileNotFoundError:
        pass

    content = None

    for root, dirs, files in os.walk("build/docs/"):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                print("Postprocessing ", file_path)
                with open(file_path, 'r') as fin:
                    content = fin.read()
                    content = content.replace(
                        '''<a href="#" class="icon icon-home"> ParlAI''',
                        '''
                        <a href="/" style="float: left">
                            <img style="padding: 0px; background-color: #fff; width: 53px; height: 53px; margin-left: 70px;" src="/static/img/icon.png">
                        </a>
                        <a href="/" style="color: #000; float: left; margin-top: 12px; font-size: 20px; font-weight: 600">
                            ParlAI
                        </a>''')
                    content = content.replace(
                        '''<a href="index.html" class="icon icon-home"> ParlAI''',
                        '''
                        <a href="/" style="float: left">
                            <img style="padding: 0px; background-color: #fff; width: 53px; height: 53px; margin-left: 70px;" src="/static/img/icon.png">
                        </a>
                        <a href="/" style="color: #000; float: left; margin-top: 12px; font-size: 20px; font-weight: 600">
                            ParlAI
                        </a>''')
                with open(file_path, 'w') as fout:
                    fout.write(content)

