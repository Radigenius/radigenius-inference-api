# cd /workspace/

# # Clone the repository
# echo "Cloning the repository..."
# git clone git@github.com:Radigenius/radigenius-inference-api.git

cd /workspace/radigenius-inference-api

# updating the os
echo "Updating the os..."
apt-get update
apt-get upgrade -y


# installing virtualenv
echo "Installing virtualenv..."
apt-get install virtualenv -y

# creating virtualenv
echo "Creating virtualenv..."
python3 -m venv venv

# call Run.sh
echo "Calling Run.sh..."
sh ./scripts/run.sh
