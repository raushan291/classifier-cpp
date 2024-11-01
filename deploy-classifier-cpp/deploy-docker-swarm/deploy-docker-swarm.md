# Docker Swarm Deployment:

1. Initialize docker swarm
```bash
    $ docker swarm init
        or 
    $ docker swarm init --advertise-addr 192.168.1.6
```

2. Deploy the application
```bash
    $ docker service create --name cpp-flask-app --replicas 3 -p 5000:5000 raushan0291/cpp-classifier-flask

    # If you want to add the exposed port then run: $ docker service update --publish-add 5000:5000 cpp-flask-app
```

3. Verify the deployment
```bash
    $ docker service ls
    $ docker service ps cpp-flask-app
```

4. Clean up
```bash
    $ docker service rm cpp-flask-app
    $ docker rmi raushan0291/cpp-classifier-flask
```
