all: clean
	python generate.py
	cd ../docs; make html
	pwd
	cp -a ../docs/build/html build/docs
	cp -a static build/
	python postprocess_docs.py

clean:
	rm -rf build/*

