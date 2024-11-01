# Kubernetes Deployment:

1. Use kubectl to deploy your application to a Kubernetes cluster.
```bash
    $ kubectl apply -f deployment.yml
    $ kubectl apply -f service.yml
```

2. Check the status of your deployment and service:
```bash
    $ kubectl get deployments
    $ kubectl get services
```

3. To view the logs of your pods:
```bash
    $ kubectl logs -l app=cpp-classifier-flask-app
```

4. To describe services
```bash
    $ kubectl describe services cpp-classifier-flask-service
```

5. To access dashboard
```bash
    $ minikube dashboard
```

6. To access your application, get the external IP of your service:
```bash
    $ kubectl get service cpp-classifier-flask-service
```

7. To get minikube ip
```bash
    $ minikube ip
```

8. To access your application
```bash
    $ http://<minikube ip>:<port> for e.g, http://192.168.49.2:<port>/
```

9. To delete service and deployment
```bash
    $ kubectl delete service cpp-classifier-flask-service
    $ kubectl delete deployment cpp-classifier-flask-app
```

10. Troubleshooting
```bash
    $ minikube delete (to delete minikube)
    $ minikube start
    $ minikube status
```
