docker build -t sketch-recognition:latest .

docker run -p 8501:8501 -p 8000:8000 sketch-recognition:latest

modal deploy model-deploy.py

also for login to wandb we should first execute 
> wandb login

similarly for Modal
install modal (pip install modal)

modal run model-deploy.py::main --action=setup
