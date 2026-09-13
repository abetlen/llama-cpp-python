# llama-cpp-python Helm Chart

This is a simple Helm Chart that simplifies the deployment of llama-cpp-python on Kubernetes.

## Features
- Utilizes the [helmet](https://github.com/companyinfo/helm-charts/tree/main/charts/helmet) library chart
- Uses the same Docker image
- Downloads models through an [init-container](./values.yaml)
- Offers a wide range of [configuration possibilities](https://github.com/companyinfo/helm-charts/tree/main/charts/helmet#parameters)

If you have any questions about the chart, please create an issue and mention @3deep5me.