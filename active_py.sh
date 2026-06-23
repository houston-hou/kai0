export VIRTUAL_ENV=$HOME/kai0/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

which python
python -c "import sys; print(sys.executable); print(sys.prefix)"
