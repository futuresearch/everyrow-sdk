from typing import Literal

import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel, Field

DBPEDIA_LABEL_NAMES = {
    0: "Company",
    1: "Educational Institution",
    2: "Artist",
    3: "Athlete",
    4: "Office Holder",
    5: "Mean Of Transportation",
    6: "Building",
    7: "Natural Place",
    8: "Village",
    9: "Animal",
    10: "Plant",
    11: "Album",
    12: "Film",
    13: "Written Work",
}
DBPEDIA_CATEGORY_TO_ID = {v: k for k, v in DBPEDIA_LABEL_NAMES.items()}


class DBpediaClassification(BaseModel):
    category: Literal[
        "Company",
        "Educational Institution",
        "Artist",
        "Athlete",
        "Office Holder",
        "Mean Of Transportation",
        "Building",
        "Natural Place",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Written Work",
    ] = Field(description="The DBpedia ontology category")


DBPEDIA_ORACLE_TASK = """Classify this text into exactly one DBpedia ontology category:
- Company: Business organizations, corporations
- Educational Institution: Schools, universities, colleges
- Artist: Musicians, painters, actors, performers
- Athlete: Sports players, competitors
- Office Holder: Politicians, government officials
- Mean Of Transportation: Vehicles, aircraft, ships
- Building: Structures, landmarks, architectural works
- Natural Place: Geographic features, parks, bodies of water
- Village: Small settlements, towns
- Animal: Species of animals
- Plant: Species of plants, flowers, trees
- Album: Music albums, discographies
- Film: Movies, documentaries
- Written Work: Books, articles, publications

Return the most appropriate category based on the content."""


class DatasetConfig(BaseModel):
    name: str
    hf_dataset: str
    text_column: str
    label_column: str
    label_names: dict[int, str]
    category_to_id: dict[str, int]
    response_model: type[BaseModel]
    oracle_task: str
    train_split: str = "train"

    # Stratified sampling fraction (e.g., 0.1 for 10%)
    sample_fraction: float | None = None

    def load(self, random_state: int = 42) -> pd.DataFrame:
        """Load the dataset from HuggingFace.

        Only loads the train split - the experiment runner handles train/test splitting.

        Args:
            random_state: Random state for stratified sampling (if sample_fraction is set)

        Returns:
            DataFrame with columns: text, label, label_name
        """
        print(f"Loading {self.name} dataset...")
        dataset = load_dataset(self.hf_dataset)

        data = dataset[self.train_split]
        df = pd.DataFrame(
            {
                "text": data[self.text_column],
                "label": data[self.label_column],
            }
        )
        df["label_name"] = df["label"].map(self.label_names)

        # Apply stratified sampling if configured
        if self.sample_fraction is not None:
            sampled = []
            for _, group in df.groupby("label"):
                sampled.append(
                    group.sample(frac=self.sample_fraction, random_state=random_state)
                )
            df = pd.concat(sampled).reset_index(drop=True)

        print(f"Loaded {len(df)} samples")
        print(f"Labels: {list(self.label_names.values())}")

        return df


DATASETS: dict[str, DatasetConfig] = {
    "dbpedia": DatasetConfig(
        name="dbpedia",
        hf_dataset="fancyzhx/dbpedia_14",
        text_column="content",
        label_column="label",
        label_names=DBPEDIA_LABEL_NAMES,
        category_to_id=DBPEDIA_CATEGORY_TO_ID,
        response_model=DBpediaClassification,
        oracle_task=DBPEDIA_ORACLE_TASK,
    ),
    "dbpedia_tiny": DatasetConfig(
        name="dbpedia_tiny",
        hf_dataset="fancyzhx/dbpedia_14",
        text_column="content",
        label_column="label",
        label_names=DBPEDIA_LABEL_NAMES,
        category_to_id=DBPEDIA_CATEGORY_TO_ID,
        response_model=DBpediaClassification,
        oracle_task=DBPEDIA_ORACLE_TASK,
        sample_fraction=0.05,
    ),
}


def get_dataset(name: str) -> DatasetConfig:
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASETS[name]
