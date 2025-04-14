cd /workspace/radigenius-inference-api

# activating the virtualenv
echo "Activating the virtualenv..."
source venv/bin/activate

# running the server
echo "Running the server..."
python -m app.main run-server
