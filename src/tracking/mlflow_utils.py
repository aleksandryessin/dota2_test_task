import mlflow


def setup_experiment(experiment_name="dota2-first-pick", tracking_uri="mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_run(model_name, params, metrics):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", model_name)
