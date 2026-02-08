# Use system uv if available, otherwise install locally
UV_SYSTEM:=$(shell command -v uv 2>/dev/null)
UV_BIN:=$(if $(UV_SYSTEM),$(UV_SYSTEM),$(CURDIR)/.uv/uv)
UV_VENV:=$(CURDIR)/.venv
UV_DEPS:=$(UV_VENV)/.deps

.PHONY: help
help: # with thanks to Ben Rady
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

$(CURDIR)/.uv/uv:
	@echo "Installing uv..."
	@mkdir -p $(dir $@)
	@curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR=$(CURDIR)/.uv sh -s

# Only require local uv installation if system uv is not available
# When UV_SYSTEM is set, UV_BIN points to the system uv, so no dependency needed
# When UV_SYSTEM is empty, UV_BIN points to .uv/uv, but we don't want a circular dependency
ifneq ($(UV_SYSTEM),)
$(UV_BIN):
	@true
endif

.PHONY: clean
clean:  ## Cleans up everything
	rm -rf $(CURDIR)/.uv $(UV_VENV) uv.lock

.PHONY: deps
deps: $(UV_BIN) $(UV_DEPS)  ## Installs and configures dependencies

$(UV_DEPS): $(UV_BIN) pyproject.toml
	$(UV_BIN) sync --no-install-project
	@touch $@
