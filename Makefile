.PHONY: cpp

install:
	@uv pip install --verbose ./python/

uninstall:
	@uv pip -v uninstall kiss_icp

editable:
	@uv pip install scikit-build-core pyproject_metadata pathspec pybind11 ninja cmake
	@uv pip install -ve ./python/

test:
	@python -m pytest -rA --verbose ./python/

cpp:
	@cmake -Bbuild cpp/kiss_icp/
	@cmake --build build -j$(nproc --all)
