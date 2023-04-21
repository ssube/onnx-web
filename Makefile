include $(shell find $(ROOT_PATH) -mindepth 2 -maxdepth 2 -name 'Makefile' | grep -v -e node_modules -e site-packages)
