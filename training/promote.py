from mlflow.tracking import MlflowClient
c = MlflowClient("http://mlflow:5000")
vs = c.search_model_versions("name='deepvision-classifier'")
latest = max(vs, key=lambda v: int(v.version))
c.transition_model_version_stage("deepvision-classifier", latest.version, "Production", True)
print(f"✅ v{latest.version} → Production")
