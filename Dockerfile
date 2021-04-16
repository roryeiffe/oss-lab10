FROM python:3.5
RUN apt-get update
RUN pip install python-tk 
RUN pip install xterm 
RUN pip install x11-apps 
RUN pip install qt5-default
RUN pip install matplotlib PyQt5
RUN pip install Flask

ADD . /opt/lab10

WORKDIR /opt/lab10

# Expose the application's port
EXPOSE 5000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]