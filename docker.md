docker run -it -p 8888:8888 -e "DISPLAY"=host.docker.internal:0 tensorflow/tensorflow:latest
apt-get update
apt-get install python-tk xterm x11-apps qt5-default
xeyes & # Just a test to make sure our display is working
pip install matplotlib PyQt5