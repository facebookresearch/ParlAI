# Updating the documentation

First install the theme we use:
```
pip install -r requirements.txt
```

Then use make to build the html:
```
make html
```

Then, make any changes to the documentation in `source`, do `make html` again, and check your changes by checking the html files in `build/html` (you can start by opening `index.html`).
If everything looks good, make a Pull Request and the documentation will update automatically once the change has landed.
If the documentation is not up to date within an hour after landing, please file an issue.
