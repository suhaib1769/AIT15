ARG PARENT_IMAGE
FROM $PARENT_IMAGE

RUN apt update && \ 
    apt install swig xvfb freeglut3-dev ffmpeg -y 

RUN pip install gym[box2d]==0.20 torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu

CMD /bin/bash