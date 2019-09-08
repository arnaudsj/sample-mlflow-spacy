include .env
export

run:
	mlflow run .

clean:
	rm -Rf ./mlruns