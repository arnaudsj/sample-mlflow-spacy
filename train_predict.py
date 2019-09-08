import os
import click
import tempfile

import pandas as pd
import mlflow
import mlflow.pyfunc

import spacy
from spacy_langdetect import LanguageDetector


class SpacyLangDetector(mlflow.pyfunc.PythonModel):

    def __init__(self):
        self.nlp = None

    def predict(self, context, model_input):
        if self.nlp is None:
            self.nlp = spacy.load('en_core_web_sm')
            self.nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

        return model_input[model_input.columns[0]].apply(lambda x: self.nlp(x)._.language)


if __name__=="__main__":
    with mlflow.start_run():
        spacyLangDetector = SpacyLangDetector()
        mlflow.pyfunc.log_model(artifact_path="model", conda_env="conda.yaml",
                                python_model=spacyLangDetector)

        pyfunc_model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id,
            artifact_path="model")
        loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_uri)

        model_input = pd.DataFrame([{"text": u"This is an english text."},
                                    {"text":u"Ce texte est en français"}])
        model_output = loaded_pyfunc_model.predict(model_input)
        print(model_output)
