ParlAI Docker image
==================

**Authors**: Mojtaba Komeili

We offer a Docker image that provides a ready to use environment for ParlAI.
This image comes with pre-installed ParlAI, along with some extra packages often required for using it.
You can find it under Packages in our main Github repository ([link](https://github.com/orgs/facebookresearch/packages?repo_name=ParlAI)).
Using this image, you can run ParlAI *anywhere* that you can run Docker, without having to install ParlAI or worry about its dependencies.

## Pulling the image
To try the latest version of the package simply run
```console
docker run ghcr.io/facebookresearch/parlai:latest
```
The default command in the Docker image is `parlai party`.
With that, you will see the dancing parrot after you run the image on its default command (as we did here).

### Running single ParlAI command
You can override the default image command to any other ParlAI (in fact container OS) command by appending it after the above command:
```console
docker run ghcr.io/facebookresearch/parlai:latest \
<PARLAI_COMMAND>
```
Here, `<PARLAI_COMMAND>` is the command that you want to run.
For example, if you want to run `parlai display_data -t wizard_of_wikipedia`,
then you will run
```console
docker run ghcr.io/facebookresearch/parlai:latest \
parlai display_data -t wizard_of_wikipedia
```

### Interactive ParlAI shell
The use case explained above only runs a single ParlAI command.
A better usage can be running an interactive terminal with the environment provided by this image.
The following command will give you a shell inside the container:
```console
docker run -it ghcr.io/facebookresearch/parlai:latest bash
```
Once in this terminal you can run any ParlAI command and have interactions with ParlAI as you have it installed on your machine.
After you are done you may exit this terminal as usual (`exit` command or <kbd>Ctrl</kbd> + <kbd>D</kbd>).

## Runtime resources
Running the interactive version of the image, as we did above, will give you the default runtime resources in your container
(if you have Docker dashboard installed, you can configure this via `Preferences` setting).
Depending on what you want to do, this may not be enough.
For example, most likely the default setting will not include the GPU resources you may need;
so, if you are inside the terminal of the image from the command above,
trying `python -c 'import torch;print(torch.cuda.is_available())'` will return `False`, regardless of the number of GPUs you have on your machine.

The solution is to add extra resources (eg, RAM, GPU etc.) to your runtime environment.
For example, the following command will give your runtime environment 2 GPUs
(the ones with 0 and 3 ids) if your machine has them at its disposal:
```console
docker run -it --gpus 0,3 ghcr.io/facebookresearch/parlai:latest bash
```
For further details on setting other resources (CPU, memory, etc.) consult [Docker documentation](https://docs.docker.com/config/containers/resource_constraints/).

## Persisting the data
Using the image as we did so far, does not persist anything that you do in the container:
all your data and code changes will be lost the next time you run the image.
In order to persist your environment, you can use Docker volumes.

First create a Docker volume: `docker volume create parlai` will create a Docker volume called *parlai*.
Now, you can mount this volume to a directory on your runtime container.
For example this will add `/data` directory inside your container runtime:
```console
docker run -it --mount source=parlai,target=/data ghcr.io/facebookresearch/parlai:latest bash
```
Anything (model, data dump, etc.) that is stored inside the `/data` directory is persisted and you can use it the next time you run this image *if* you mount your Docker volume again.

### Changing ParlAI code
You can even persist the code inside your ParlAI codebase on the Volume.
This way you can develop custom code in ParlAI and keep your changes between sessions of running your container.
All you need to do for this is to set the target of your mounted volume on where ParlAI currently exist inside the container:
```console
docker run -it --mount source=parlai,target=/root/ParlAI ghcr.io/facebookresearch/parlai:latest bash
```
:::{note} Automatic Processing
Be cautious about mounting your volume on different target directories each time.
Switching a volume between targets may corrupt your volume data.
In case you ended up with a corrupted volume, you may simply create a new volume and start over
(but your previously persisted data might be unrecoverable).
:::
