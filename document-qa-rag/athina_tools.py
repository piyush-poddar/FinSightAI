from athina_client.datasets import Dataset
from athina_client.keys import AthinaApiKey

AthinaApiKey.set_key('ATHINA_API_KEY')


def upload_dataset(name, rows):
    try:
        dataset = Dataset.create(
            name=name,
            # All fields below are optional
            rows=rows
            # project_name="project_name", # Note: project name should already exist on Athina
        )
    except Exception as e:
        print(f"Failed to create dataset: {e}")

