from huggingface_hub import HfApi
api = HfApi()
repo_id = "pancakewithoutsleep/phishing-detector"

# create repo if it doesn't exist (private)
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

# upload_folder will use your cached token, no need to paste it here
api.upload_folder(
    repo_id=repo_id,
    folder_path="phishing_model_deployment/model",
    path_in_repo="",
    repo_type="model",
    commit_message="Upload trained DistilBERT model"
)
print("Upload finished")
