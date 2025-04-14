cd /workspace/radigenius-inference-api

# pulling the latest changes
echo "Pulling the latest changes..."
git pull origin main

# activating the virtualenv
echo "Activating the virtualenv..."
source venv/bin/activate

# installing the dependencies
echo "Installing the dependencies..."
pip install -r requirements.txt

# running the server
echo "Running the server..."
python -m app.main run-server
