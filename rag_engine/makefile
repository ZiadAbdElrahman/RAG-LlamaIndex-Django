
COMFY_DOCKER_IMAGE ?= rag
DOCKER_TAG ?= develop
DEV_DOCKER_NAME=rag_gpu_dev
DOCKER_COMFY_NAME=rag

build_cpu:
	docker build -t ${DOCKER_COMFY_NAME}:${DOCKER_TAG} -f Dockerfile.cpu .


build_gpu:
	docker build -t ${DOCKER_COMFY_NAME}:${DOCKER_TAG} -f Dockerfile.gpu .

run_cpu:
	docker stop ${COMFY_DOCKER_IMAGE} || echo "ok"
	docker rm ${COMFY_DOCKER_IMAGE} || echo "ok"
	docker run -it --gpus all --shm-size=16GB \
			-e ENV \
			--name ${COMFY_DOCKER_IMAGE} \
			--volume ./:/app \
			--volume ./models:/app/models \
			${DOCKER_COMFY_NAME}:${DOCKER_TAG}
			
run_gpu:
	docker stop ${COMFY_DOCKER_IMAGE} || echo "ok"
	docker rm ${COMFY_DOCKER_IMAGE} || echo "ok"
	docker run -it --gpus all --shm-size=16GB \
			-e ENV \
			--name ${COMFY_DOCKER_IMAGE} \
			--volume ./:/app \
			--volume ./models:/app/models \
			${DOCKER_COMFY_NAME}:${DOCKER_TAG}
