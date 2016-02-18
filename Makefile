
PROG := python3 setup.py

.PHONY : clean build install test

test:
	nosetests-3.4 -v

clean:
	$(PROG) clean

build:
	$(PROG) build

install:
	$(PROG) install