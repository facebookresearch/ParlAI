all: clean
	python generate.py
	cd ../docs; make html
	pwd
	cp -a ../docs/build/html build/docs
	cp build/docs/_static/fonts/RobotoSlab/roboto-slab-v7-regular.eot build/docs/_static/fonts/RobotoSlab/roboto-slab.eot
	cp -a static build/
	python postprocess_docs.py

clean:
	rm -rf build/*

