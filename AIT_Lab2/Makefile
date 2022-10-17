PARENT_IMAGE=python:3.8.14-bullseye
IMAGE_NAME=ait-rl-env
TAG=latest

build: clean
	docker build -f ./docker/Dockerfile -t $(IMAGE_NAME):$(TAG)\
			--build-arg PARENT_IMAGE=$(PARENT_IMAGE) .
enter:
	docker run \
			--mount type=bind,source="$(shell pwd)",target=/workspace \
			--workdir "/workspace" \
			-it --rm $(IMAGE_NAME):$(TAG) \
			/bin/bash

run_dqn:
	docker run \
			--mount type=bind,source="$(shell pwd)",target=/workspace \
			--workdir "/workspace" \
			-it --rm $(IMAGE_NAME):$(TAG) \
			/bin/bash -c "Xvfb :1 -screen 0 1024x768x16 > /dev/null 2>&1 & export DISPLAY=:1 && python3 deep_q_learning_main.py"

run_ql:
	docker run \
			--mount type=bind,source="$(shell pwd)",target=/workspace \
			--workdir "/workspace" \
			-it --rm $(IMAGE_NAME):$(TAG) \
			python3 q_learning_main.py

clean:
	if [ -d "./__pycache__" ]; \
	then docker run \
			--mount type=bind,source="$(shell pwd)",target=/workspace \
			--workdir "/workspace" \
			-it --rm $(IMAGE_NAME):$(TAG) \
			rm -rf __pycache__/ recorded_episodes/; \
	fi

export_image:
	docker save $(IMAGE_NAME):$(TAG) | gzip > ./docker/$(IMAGE_NAME).tar.gz

import_image:
	docker load < ./docker/$(IMAGE_NAME).tar.gz

.SILENT: clean build run_dqn run_ql enter