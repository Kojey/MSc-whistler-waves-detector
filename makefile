IPY=./ipy
LIB=./lib
PY=./py
DEP=parameters.py
DEPS= $(PY)/$(DEP)

$(PY)/%.py: $(LIB)/%.ipynb $(DEPS)
	jupyter nbconvert --to python --ouptput-dir $(PY) $(LIB)/%.ipynb

python: $(LIB)/*.ipynb $(IPY)/*.ipynb
	jupyter nbconvert --to python --output-dir $(PY) $(LIB)/*.ipynb
	jupyter nbconvert --to python --output-dir $(PY) $(IPY)/*.ipynb

clean:
	rm $(PY)/*.py
