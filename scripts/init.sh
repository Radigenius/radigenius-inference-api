cd /workspace/

# Clone the repository
echo "Cloning the repository..."
git clone git@github.com:Radigenius/radigenius-inference-api.git

cd radigenius-inference-api

# updating the os
echo "Updating the os..."
sudo apt-get update
sudo apt-get upgrade -y


# installing virtualenv
echo "Installing virtualenv..."
sudo apt-get install virtualenv -y

# creating virtualenv
echo "Creating virtualenv..."
python3 -m venv venv

# activating the virtualenv
echo "Activating the virtualenv..."
source venv/bin/activate

# installing the dependencies
echo "Installing the dependencies..."
pip install -r requirements.txt

# running the server
echo "Running the server..."
python -m app.main run-server
