
PROG := python setup.py

.PHONY : clean build install test

test:
	$(PROG) test

clean:
	$(PROG) clean

build:
	$(PROG) build

install:
	$(PROG) install