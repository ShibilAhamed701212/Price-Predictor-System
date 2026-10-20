import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    """Run the prices predictor deployment pipeline"""
    # Get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Services started by the deployment step. They are registered under MLflow's model
    # name ("model"), not "prices_predictor", so they are looked up by pipeline and step.
    existing_services = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
    )

    if stop_service:
        for service in existing_services:
            if service.is_running:
                service.stop(timeout=10)
        return

    # Remove previous deployments first: otherwise the deployer step matches the old service
    # and keeps serving the first model ever deployed instead of the newly trained one.
    for service in existing_services:
        model_deployer.delete_model_server(service.uuid, timeout=10)

    # Run the continuous deployment pipeline
    continuous_deployment_pipeline()

    # Run the inference pipeline
    inference_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `continuous_deployment_pipeline` "
        "experiment."
    )

    # Fetch existing services with the same pipeline name, step name, and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
    )

    if service:
        print(
            f"The MLflow prediction server is running locally as a daemon "
            f"process and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


if __name__ == "__main__":
    run_main()
