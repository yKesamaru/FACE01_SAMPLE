

docker build -t face01_gpu:1.4.10 -f docker/Dockerfile_gpu . --network host
docker build -t face01_no_gpu:1.4.10 -f docker/Dockerfile_no_gpu . --network host