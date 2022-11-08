ParlAI Docker image
==================

**Authors**: Mojtaba Komeili

We offers a Docker image that provides a ready to use environment for ParlAI.
This image comes with ParlAI, along with some extra packages often required for using it, pre-installed.
You can find it under Packages in our main Github repostiry [link](https://github.com/orgs/facebookresearch/packages?repo_name=ParlAI).
Using this image you can run ParlAI *anywhere* that you can have Docker without having to install it or its dependancies.

## Pulling the image
To try the latest version of the package simply run this:
```
docker run ghcr.io/facebookresearch/parlai:latest
```
The default command in the Docker image is `parlai party`.
So you will see the dancing parrot well you run the image on its default command, as we did above.

### Running single ParlAI command
You can run any other ParlAI command liek this
```
docker run ghcr.io/facebookresearch/parlai:latest \
<PARLAI_COMMAND>
```
replacing your `<PARLAI_COMMAND>` with the command that you want to run.
For example, if you want to run `parlai display_data -t wizard_of_wikipedia`,
then you have
```
docker run ghcr.io/facebookresearch/parlai:latest \
parlai display_data -t wizard_of_wikipedia
```

### Interactive ParlAI shell
The use case explained above only runs a single ParlAI command.


## Coding in ParlAI


## Setting the requirements


## Persisting the data
